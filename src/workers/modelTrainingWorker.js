import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
const _globalCtx = {};

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1

}

const normalize = ( valor, min, max ) => (valor - min) / (max - min) || 1;

function makeContext(catalog, users) {
    const ages = users.map(u => u.age);
    const prices = catalog.map(p => p.price);

    const maxAge = Math.max(...ages);
    const minAge = Math.min(...ages);

    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);

    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];

    const colorIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    );
    const categoryIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    );
        
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        (user.purchases.forEach(purchase => {
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age;
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1;
        }))       
    });

    const productAvgAgesNorm = Object.fromEntries(
        catalog.map(product => {
            const avg = ageCounts[product.name] ? 
                ageSums[product.name] / ageCounts[product.name] : midAge;
            return [product.name, normalize(avg, minAge, maxAge)];
        })
    );

    return {
        catalog,
        users,
        colorIndex,
        categoryIndex,
        productAvgAgesNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCatagories: categories.length,
        numColors: colors.length,
        dimentions: 2 + categories.length +  colors.length // price + age + color + category
        
    }
    

}

const oneHotWeight = (index, length, weight) => {
    return tf.oneHot(index ?? 0, length).cast('float32').mul(weight);
}
function encodeProcurct(product, context) {
    const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice)]).mul(WEIGHTS.price);
    const age = tf.tensor1d([(context.productAvgAgesNorm[product.name] ?? 0.5)]).mul(WEIGHTS.age);
    const color = oneHotWeight(context.colorIndex[product.color], context.numColors, WEIGHTS.color);
    const category = oneHotWeight(context.categoryIndex[product.category], context.numCatagories, WEIGHTS.category);
    
    return tf.concat1d([price, age, category,color]);
}


async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await (await fetch('../../data/products.json')).json();
    console.log('Catalog loaded:', catalog)
    const context = makeContext(catalog, users)

    context.productVectors = catalog.map(product => {
        return {
            name: product.name,
            meta: { ...product},
            vector: encodeProcurct(product, context).dataSync()
        }
    });
    console.log('Product vectors created:', context.productVectors)
     debugger
    _globalCtx = context;

   

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
