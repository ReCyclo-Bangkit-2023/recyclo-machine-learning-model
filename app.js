// use node version 10.16.3
// and tfjs-node version 1.2.11

// Include tensorflow module
const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

const priceDict = {
  0: 1500,
  1: 12000,
  2: 2000,
  3: 3000,
};

async function loadModel(path) {
  // Get file path
  const handler = tfnode.io.fileSystem(path);

  // Loda model from file path
  const model = await tf.loadLayersModel(handler);

  return model;
}

async function loadImage(imageBuffer) {
  // Convert buffer imgage into tensor
  const tensor = tfnode.node.decodeImage(imageBuffer);

  // Resize image to fit the model
  const resizedTensor = tf.image
    .resizeBilinear(tensor, [150, 150])
    .div(tf.scalar(255));
  const expandedTensor = resizedTensor.expandDims(0);

  return expandedTensor;
}

async function predict(imageBuffer) {
  // Load model
  const modelClass = await loadModel(
    './object-classification-model/model.json'
  );
  const modelFeature = await loadModel('./object-features-model/model.json');

  // load image
  const img = await loadImage(imageBuffer);

  // Doing prediction
  const predictClass = modelClass.predict(img);
  const predictFfeature = modelFeature.predict(img);

  // Print prediction result
  // Prediction result return array type data

  // Do price prediction from object class and feature
  objectClass = predictClass.argMax(1).arraySync()[0];
  objectFeature = predictFfeature.arraySync()[0][0];
  priceResult = priceDict[objectClass] * objectFeature;

  return Math.ceil(priceResult);
}

const initHapiServer = async () => {
  const hapiServer = Hapi.server({
    port: 9000,
    host: process.env.NODE_ENV !== 'production' ? 'localhost' : '0.0.0.0',
  });

  hapiServer.route([
    {
      method: 'POST',
      path: '/api/recommendation-price',
      handler: async (request, h) => {
        try {
          const { image } = request.payload;

          const priceRecommendation = await predict(image);

          return h.response({
            error: true,
            message: 'success',
            data: {
              priceRecommendation,
            },
          });
        } catch (error) {
          return h.response({
            error: true,
            message: error.message,
            data: {},
          });
        }
      },
      options: {
        payload: {
          parse: true,
          allow: 'multipart/form-data',
          multipart: { output: 'data' },
          maxBytes: 8388608,
        },
      },
    },
  ]);

  await hapiServer.start();

  console.info(`Server is running on ${hapiServer.info.uri}`);
};

initHapiServer();
