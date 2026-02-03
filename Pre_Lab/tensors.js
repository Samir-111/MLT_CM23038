// Import TensorFlow.js
const tf = require('@tensorflow/tfjs');

// Scalar (0-D Tensor)
const scalar = tf.scalar(5);
console.log("Scalar (0-D Tensor):");
scalar.print();

// Vector (1-D Tensor)
const vector = tf.tensor1d([1, 2, 3]);
console.log("\nVector (1-D Tensor):");
vector.print();

// Matrix (2-D Tensor)
const matrix = tf.tensor2d([
  [1, 2],
  [3, 4]
]);
console.log("\nMatrix (2-D Tensor):");
matrix.print();
