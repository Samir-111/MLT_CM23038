async function assignment3() {

  // Training data
  const x = tf.tensor1d([1, 2, 3, 4, 5]);
  const y = tf.tensor1d([2, 4, 6, 8, 10]);

  // Model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  await model.fit(x, y, { epochs: 200 });

  // Unseen inputs
  const unseenInputs = tf.tensor1d([6, 7, 8]);
  const predictions = model.predict(unseenInputs);

  console.log("Unseen Inputs:");
  unseenInputs.print();

  console.log("Predicted Outputs:");
  predictions.print();
}

assignment3();
