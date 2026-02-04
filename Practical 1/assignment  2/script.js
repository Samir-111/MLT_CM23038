async function assignment1() {

  // Synthetic data (y = 2x)
  const x = tf.tensor1d([1, 2, 3, 4, 5]);
  const y = tf.tensor1d([2, 4, 6, 8, 10]);

  // Model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  // Training
  await model.fit(x, y, { epochs: 150 });

  // Prediction
  const predictions = model.predict(x);

  console.log("Actual Values:");
  y.print();

  console.log("Predicted Values:");
  predictions.print();
}

assignment1();
