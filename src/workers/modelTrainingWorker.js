// Importa TensorFlow.js para o worker (necessário para processamento em background)
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

// Constantes para eventos de comunicação entre worker e thread principal
// Permite enviar mensagens assíncronas sobre progresso, logs, etc.
const workerEvents = {
    trainingComplete: 'training:complete',  // Treinamento finalizado
    trainModel: 'train:model',              // Comando para iniciar treinamento
    recommend: 'recommend',                 // Comando para recomendações
    trainingLog: 'training:log',            // Logs de cada época
    progressUpdate: 'progress:update',      // Atualização de progresso
    tfVisData: 'tfvis:data',                // Dados para visualização (não usado)
    tfVisLogs: 'tfvis:logs',                // Logs para visualização (não usado)
};

// Log inicial para confirmar que o worker foi carregado
console.log('Model training worker initialized');

// Contexto global para armazenar dados processados (catálogo, usuários, índices)
let _globalCtx = {};

// Modelo treinado global
let _model = null;

// Pesos para diferentes features no vetor de produto
// Valores entre 0 e 1 que determinam a importância de cada feature
const WEIGHTS = {
    category: 0.4,  // Peso para categoria (ex.: eletrônico)
    color: 0.3,     // Peso para cor (ex.: preto)
    price: 0.2,     // Peso para preço
    age: 0.1        // Peso para idade média dos compradores
}

// Função utilitária para normalizar valores entre 0 e 1
// Exemplo: normalize(25, 20, 40) = (25-20)/(40-20) = 0.25
const normalize = (valor, min, max) => (valor - min) / (max - min) || 1;

// Função principal para criar o contexto de dados
// Processa catálogo e usuários para criar índices, estatísticas, etc.
// Entrada: catalog (array de produtos), users (array de usuários)
// Saída: objeto com todos os dados processados necessários para ML
function makeContext(catalog, users) {
    // Extrai idades e preços para calcular min/max
    const ages = users.map(u => u.age);           // [25, 30, 28, ...]
    const prices = catalog.map(p => p.price);     // [129.99, 199.99, ...]

    // Calcula estatísticas básicas
    const maxAge = Math.max(...ages);     // Ex.: 40
    const minAge = Math.min(...ages);     // Ex.: 20
    const maxPrice = Math.max(...prices); // Ex.: 199.99
    const minPrice = Math.min(...prices); // Ex.: 39.99

    // Cria listas únicas de cores e categorias
    const colors = [...new Set(catalog.map(p => p.color))];         // ['preto', 'azul', ...]
    const categories = [...new Set(catalog.map(p => p.category))];  // ['eletrônico', 'vestuário', ...]

    // Cria mapas cor -> índice para one-hot encoding
    // Ex.: { 'preto': 0, 'azul': 1, 'branco': 2 }
    const colorIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    );

    // Mesmo para categorias
    const categoryIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    );

    // Idade média geral (fallback)
    const midAge = (minAge + maxAge) / 2;

    // Calcula idade média por produto baseada em quem comprou
    const ageSums = {};   // Soma das idades por produto
    const ageCounts = {}; // Contagem de compras por produto

    // Para cada usuário e suas compras, acumula idades
    users.forEach(user => {
        user.purchases.forEach(purchase => {
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age;
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1;
        });
    });

    // Calcula idade média normalizada por produto
    // Ex.: produto 'Fones' comprado por usuários de 25 e 30 -> avg = 27.5 -> normalizado
    const productAvgAgesNorm = Object.fromEntries(
        catalog.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] : midAge;
            return [product.name, normalize(avg, minAge, maxAge)];
        })
    );

    // Retorna contexto completo
    return {
        catalog,           // Lista de produtos
        users,             // Lista de usuários
        colorIndex,        // Mapa cor -> índice
        categoryIndex,     // Mapa categoria -> índice
        productAvgAgesNorm,// Idade média normalizada por produto
        minAge, maxAge,    // Faixa de idades
        minPrice, maxPrice,// Faixa de preços
        numCatagories: categories.length, // Número de categorias únicas
        numColors: colors.length,         // Número de cores únicas
        dimensions: 2 + categories.length + colors.length // Dimensão total do vetor (preço + idade + cores + categorias)
    };
}

// Função auxiliar para criar vetor one-hot com peso
// Ex.: oneHotWeight(1, 3, 0.5) -> tensor [0, 0.5, 0] (posição 1 ativada com peso 0.5)
const oneHotWeight = (index, length, weight) => {
    return tf.oneHot(index ?? 0, length).cast('float32').mul(weight);
};

// Codifica um produto em vetor numérico para ML
// Entrada: produto (objeto), contexto (dados processados)
// Saída: tensor 1D com features normalizadas e pesadas
// Exemplo: produto {name: 'Fones', price: 129.99, color: 'preto', category: 'eletrônico'}
// -> tensor [preço_normalizado * peso, idade_avg * peso, ...one-hot color..., ...one-hot category...]
function encodeProduct(product, context) {
    // Preço normalizado e pesado
    const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice)]).mul(WEIGHTS.price);

    // Idade média dos compradores normalizada e pesada
    const age = tf.tensor1d([(context.productAvgAgesNorm[product.name] ?? 0.5)]).mul(WEIGHTS.age);

    // Cor codificada one-hot com peso
    const color = oneHotWeight(context.colorIndex[product.color], context.numColors, WEIGHTS.color);

    // Categoria codificada one-hot com peso
    const category = oneHotWeight(context.categoryIndex[product.category], context.numCatagories, WEIGHTS.category);

    // Concatena tudo em um vetor 1D
    return tf.concat1d([price, age, category, color]);
}

// Codifica um usuário baseado na média dos produtos que comprou
// Entrada: usuário (objeto com purchases), contexto
// Saída: tensor 2D [1, dimensions] com média dos vetores dos produtos comprados
// Exemplo: usuário comprou 2 produtos -> média dos 2 vetores -> [1, dimensions]
function encodeUser(user, context) {
    if (user.purchases.length) {
        // Codifica cada produto comprado
        const productTensors = user.purchases.map(
            product => encodeProduct(product, context)
        );

        // Empilha em tensor 2D e calcula média ao longo do eixo 0
        // Resultado: tensor [dimensions] com média
        const meanVector = tf.stack(productTensors).mean(0);

        // Reshape para [1, dimensions] para consistência
        return meanVector.reshape([1, context.dimensions]);
    }

    // Se o usuário não tem compras, criamos um vetor com:
    // [idade_normalizada, one-hot categorias, one-hot cores]
    return tf.concat1d([
        tf.zeros([1]),
        tf.tensor1d([
            normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
        ]), // Idade média fallback
        tf.zeros([context.numCatagories]),
        tf.zeros([context.numColors])
    ]).reshape([1, context.dimensions]);
    // Se não há compras, retorna undefined (não usado devido a filtros)
}

// Cria dados de treinamento para o modelo de recomendação
// Para cada usuário-produto, cria entrada (vetor usuário + vetor produto) e label (1 se comprou, 0 se não)
// Entrada: contexto com usuários e produtos
// Saída: {xs: tensor 2D de entradas, ys: tensor 2D de labels, inputDimensions: tamanho da entrada}
function createTraningData(context) {
    const inputs = [];  // Lista de vetores [usuário + produto]
    const labels = [];  // Lista de labels [0 ou 1]

    // Filtra usuários com compras
    context.users.filter(user => user.purchases.length > 0)
        .forEach(user => {
            // Vetor do usuário (média das compras)
            const userVector = encodeUser(user, context).dataSync();

            // Para cada produto no catálogo
            context.productVectors.forEach(product => {
                // Vetor do produto
                const productVector = encodeProduct(product, context).dataSync();

                // Label: 1 se usuário comprou este produto, 0 caso contrário
                const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;

                // Adiciona combinação usuário-produto aos dados
                inputs.push([...userVector, ...productVector]);
                labels.push(label);
            });
        });

    // Converte para tensores TensorFlow
    return {
        xs: tf.tensor2d(inputs),                           // Entradas: shape [num_amostras, 2*dimensions]
        ys: tf.tensor2d(labels, [labels.length, 1]),       // Labels: shape [num_amostras, 1]
        inputDimensions: context.dimensions * 2            // Tamanho da entrada (usuário + produto)
    };
}

// Configura e treina a rede neural
// Entrada: trainingData (xs, ys, inputDimensions)
// Saída: modelo treinado
async function configureNeuralNetAndTrain(trainingData) {
    // Cria modelo sequencial (camadas empilhadas)
    const model = tf.sequential();

    // Camada de entrada + oculta 1: 128 neurônios, ReLU
    model.add(tf.layers.dense({ inputShape: [trainingData.inputDimensions], units: 128, activation: 'relu' }));

    // Camada oculta 2: 64 neurônios, ReLU
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));

    // Camada oculta 3: 32 neurônios, ReLU
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

    // Camada de saída: 1 neurônio, sigmoid (probabilidade 0-1)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Compila modelo com otimizador Adam, loss para classificação binária
    model.compile({
        optimizer: tf.train.adam(0.01),    // Taxa de aprendizado 0.01
        loss: 'binaryCrossentropy',        // Loss para 0/1
        metrics: ['accuracy']              // Métrica de acurácia
    });

    // Treina o modelo
    await model.fit(
        trainingData.xs,    // Dados de entrada
        trainingData.ys,    // Labels
        {
            epochs: 100,        // Número de épocas
            batchSize: 32,      // Tamanho do batch
            shuffle: true,      // Embaralha dados a cada época
            callbacks: {
                // Callback para enviar logs de treinamento
                onEpochEnd: (epoch, log) => {
                    postMessage({
                        type: workerEvents.trainingLog,
                        epoch: epoch,
                        loss: log.loss,
                        accuracy: log.acc
                    });
                }
            }
        }
    );

    // Retorna modelo treinado
    return model;
}

// Função principal para treinar o modelo
// Entrada: {users} - array de usuários
// Processa dados, cria contexto, treina modelo, salva globalmente
async function trainModel({ users }) {
    console.log('Training model with users:', users);

    // Valida entrada
    if (!users || !Array.isArray(users)) {
        console.error('Users not loaded or invalid');
        postMessage({ type: workerEvents.trainingComplete });
        return;
    }

    // Filtra usuários válidos com compras
    users = users.filter(u => u && Array.isArray(u.purchases) && u.purchases.length > 0);

    // Envia progresso 50%
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    // Carrega catálogo de produtos
    const catalog = await (await fetch('../../data/products.json')).json();
    console.log('Catalog loaded:', catalog);

    // Valida catálogo
    if (!catalog || !Array.isArray(catalog)) {
        console.error('Catalog not loaded or invalid');
        postMessage({ type: workerEvents.trainingComplete });
        return;
    }

    // Cria contexto com dados processados
    const context = makeContext(catalog, users);

    // Cria vetores para produtos (com tratamento de erros)
    context.productVectors = catalog.map(product => {
        try {
            return {
                name: product.name,
                meta: { ...product },                    // Metadados do produto
                vector: encodeProduct(product, context).dataSync()  // Vetor numérico
            };
        } catch (e) {
            console.error('Error encoding product', product, e);
            return null;
        }
    }).filter(p => p !== null);  // Remove produtos com erro

    console.log('Product vectors created:', context.productVectors);

    // Salva contexto global
    _globalCtx = context;

    // Cria dados de treinamento
    const trainingData = createTraningData(context);

    // Treina o modelo
    _model = await configureNeuralNetAndTrain(trainingData);

    // Envia progresso 100% e conclusão
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

// Função para gerar recomendações (não implementada)
// Entrada: usuário e contexto
function recommend(user) {
    debugger
    if (!_model) return;
    const context = _globalCtx;

    const userTensor = encodeUser(user, context).dataSync();
    const inputs = context.productVectors.map(({ vector }) => {
        return [...userTensor, ...vector];
    });
    const inputTensor = tf.tensor2d(inputs);

    const predictions = _model.predict(inputTensor);
    const scores = predictions.dataSync();

    const recommendations = context.productVectors.map((item, index) => ({
        ...item.meta,
        name: item.name,
        score: scores[index]
    }));

    const sortedItems = recommendations.sort((a, b) => b.score - a.score);

    postMessage({
        type: workerEvents.recommend,
        recommendations: sortedItems
    });

    debugger
    console.log('will recommend for user:', user);
    // TODO: implementar lógica de recomendação usando _model
    // Exemplo: codificar usuário, prever para cada produto, ordenar por probabilidade
}

// Mapeamento de ações para funções
const handlers = {
    [workerEvents.trainModel]: trainModel,    // Handler para treinamento
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),  // Handler para recomendações
};

// Listener para mensagens da thread principal
// Recebe {action, ...data} e chama handler correspondente
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
