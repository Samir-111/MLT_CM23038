/* -------- CREATE SIMPLE MNIST-LIKE DATA -------- */
function generateData(samples = 3000) {
  console.log("Generating MNIST-like training data...");

  const xs = tf.randomNormal([samples, 28, 28, 1]);
  const labels = tf.randomUniform([samples], 0, 10, "int32");
  const ys = tf.oneHot(labels, 10);

  return { xs, ys };
}

/* -------- CNN MODEL -------- */
function buildCNN() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 16,
    kernelSize: 3,
    activation: "relu"
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

/* -------- DENSE MODEL -------- */
function buildDense() {
  const model = tf.sequential();

  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

/* -------- TRAIN & COMPARE -------- */
async function runExperiment() {

  const { xs, ys } = generateData(3000);

  /* CNN TRAINING */
  console.log("Training CNN Model...");
  const cnn = buildCNN();

  await cnn.fit(xs, ys, {
    epochs: 5,
    batchSize: 64,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(`CNN Epoch ${epoch + 1} Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
    }
  });

  const cnnEval = await cnn.evaluate(xs, ys);
  const cnnAcc = (await cnnEval[1].data())[0] * 100;
  console.log("Final CNN Accuracy:", cnnAcc.toFixed(2) + "%");


  /* DENSE TRAINING */
  console.log("\nTraining Dense Network...");
  const dense = buildDense();

  await dense.fit(xs, ys, {
    epochs: 5,
    batchSize: 64,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(`Dense Epoch ${epoch + 1} Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
    }
  });

  const denseEval = await dense.evaluate(xs, ys);
  const denseAcc = (await denseEval[1].data())[0] * 100;
  console.log("Final Dense Accuracy:", denseAcc.toFixed(2) + "%");

  /* COMPARISON RESULT */
  console.log("\n==============================");

  if (cnnAcc > denseAcc)
    console.log("Conclusion: CNN performs better than Dense Network for image data.");
  else
    console.log("Conclusion: Dense performs better (rare case).");
}

runExperiment();
