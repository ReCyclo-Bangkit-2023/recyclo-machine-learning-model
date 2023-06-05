
// use node version 10.16.3 
// and tfjs-node version 1.2.11 

const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

const handler = tfnode.io.fileSystem('./deployment/object-features-model/model.json');
const model = tfnode.loadLayersModel(handler);

// model.summary();


// async function loadModel(path){
//     const handler = tfnode.io.fileSystem(path);
//     const model = await tf.loadLayersModel(handler);
//     console.log("Model loaded");
//     return model;
// }

// async function init() {
//     const model = await loadModel('./deployment/object-features-model/model.json');
//     model.summary();
// }

// init();

// async function loadModel() {
//     const modelPath = 'file://./deployment/object-classification-model/model.json';
//     return tf.loadLayersModel(modelPath)
//         .then((model) => {
//             loadedModel = model;
//             console.log("Model loaded!");
//         })
//         .catch((error) => {
//             console.error(error);
//         });
// }

// const model = loadModel();

// console.log(model);