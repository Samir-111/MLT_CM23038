async function assignment1() {

  const xs = tf.randomNormal([500, 28, 28, 1]);
  const ys = tf.oneHot(
    tf.randomUniform([500], 0, 10, 'int32'),
    10
  );

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 8,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  console.log("Training CNN...");

  await model.fit(xs, ys, {
    epochs: 5,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} Accuracy: ${(logs.accuracy * 100).toFixed(2)}%`);
      }
    }
  });

  console.log("CNN Training Completed ✅");
}

assignment1();
