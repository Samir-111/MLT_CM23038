async function runLinearRegression() {

  // Synthetic Data (y = 2x + 1)
  const xTrain = tf.tensor1d([1, 2, 3, 4, 5]);
  const yTrain = tf.tensor1d([3, 5, 7, 9, 11]);

  console.log("Training Data:");
  xTrain.print();
  yTrain.print();

  // Create Model
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }));

  // Compile Model
  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  console.log("Training started...");

  // Train Model
  await model.fit(xTrain, yTrain, {
    epochs: 200,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 50 === 0) {
          console.log(`Epoch ${epoch} - Loss: ${logs.loss}`);
        }
      }
    }
  });

  console.log("Training completed!");

  // Prediction
  const testInput = tf.tensor1d([6]);
  const prediction = model.predict(testInput);

  console.log("Prediction for x = 6:");
  prediction.print();
}

runLinearRegression();
