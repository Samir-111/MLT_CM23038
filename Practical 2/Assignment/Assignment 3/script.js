function createFakeMNIST(samples = 2000) {
  console.log("Creating synthetic MNIST-like data...");

  const xs = tf.randomUniform([samples, 28, 28, 1]);
  const labels = tf.randomUniform([samples], 0, 10, "int32");
  const ys = tf.oneHot(labels, 10);

  return { xs, ys };
}

/* CNN MODEL */
function createCNN() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 8,
    kernelSize: 3,
    activation: "relu"
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

/* DENSE MODEL */
function createDense() {
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

/* TRAINING */
async function run() {
  const { xs, ys } = createFakeMNIST(2000);

  /* CNN */
  console.log("Training CNN...");
  const cnn = createCNN();

  await cnn.fit(xs, ys, {
    epochs: 3,
    batchSize: 64,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(`CNN Epoch ${epoch + 1} Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
    }
  });

  const cnnAcc = (await cnn.evaluate(xs, ys)[1].data())[0] * 100;
  console.log("Final CNN Accuracy:", cnnAcc.toFixed(2) + "%");

  /* Dense */
  console.log("\nTraining Dense...");
  const dense = createDense();

  await dense.fit(xs, ys, {
    epochs: 3,
    batchSize: 64,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(`Dense Epoch ${epoch + 1} Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
    }
  });

  const denseAcc = (await dense.evaluate(xs, ys)[1].data())[0] * 100;
  console.log("Final Dense Accuracy:", denseAcc.toFixed(2) + "%");

  console.log("\n======================");

  if (cnnAcc > denseAcc)
    console.log("Conclusion: CNN performed better than Dense Network");
  else
    console.log("Conclusion: Dense performed better (rare case)");
}

run();
