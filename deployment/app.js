// use node version 10.16.3 
// and tfjs-node version 1.2.11 

// Include tensorflow module
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

// Include fs module
const fs = require('fs');


async function loadModel(path) {
  // Get file path
  const handler = tfnode.io.fileSystem(path);

  // Loda model from file path
  const model = await tf.loadLayersModel(handler);

  console.log("Model loaded!");
  return model
}


async function loadImage(path) {
  // Read imgae as file buffer
  const imageBuffer = fs.readFileSync(path);

  // Convert buffer imgage into tensor
  const tensor = tfnode.node.decodeImage(imageBuffer);

  // Resize image to fit the model
  const resizedTensor  = tf.image.resizeBilinear(tensor, [150, 150]).div(tf.scalar(255));
  const expandedTensor = resizedTensor.expandDims(0);

  console.log("Image loaded!");
  return expandedTensor;
}


async function predict() {
  // Load model
  const modelClass = await loadModel('./deployment/object-classification-model/model.json');
  const modelFeature = await loadModel('./deployment/object-features-model/model.json');

  // load image
  const img = await loadImage('./deployment/plastic8.jpg');

  // Doing prediction
  const predictClass =  modelClass.predict(img);
  const predictFfeature = modelFeature.predict(img);

  // Print prediction result
  // Prediction result return array type data
  console.log(predictClass.arraySync());
  console.log(predictFfeature.arraySync());
}


predict();


