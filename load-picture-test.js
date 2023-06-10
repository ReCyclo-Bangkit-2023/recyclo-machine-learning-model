// const Jimp = require('jimp');

// Jimp.read('./deployment/DSC02781.JPG')
//   .then(image => {
//     console.log('Image loaded!');
//     // Further operations with the loaded image can be performed here
//   })
//   .catch(error => {
//     console.log('Error loading image.');
//   });




// Using file buffer

// Include fs module
const fs = require('fs');

const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

const imageBuffer = fs.readFileSync('./deployment/DSC02781.JPG');

const tensor = tfnode.node.decodeImage(imageBuffer)

const img = tf.image.resizeBilinear(tensor, [150, 150]);

console.log(img);



// Using Jimp

async function readImage(path) {
  Jimp.read(path)
  .then(image => {
    console.log('Image loaded!');
    image.resize(150,150)
    // Convert the image to a Buffer
    // image = tf.browser.fromPixels(image)

    const buffer = image.getBufferAsync(Jimp.MIME_JPEG);

    const tensor = tf.node.decodeImage(buffer)

    tf.image.resizeBilinear(tensor, [224, 224]);

    tensor.expandDims(0);

    return tensor.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  })
  .catch(error => {
    console.log('Error loading image.\n' + error);
  });
}