async function run() {

  console.log("Loading Model...");

  // Training Data
  const sentences = [
    "I love this product",
    "This is amazing",
    "Very good and nice",
    "I am very happy",
    "I hate this",
    "This is bad",
    "Very poor quality",
    "I am sad"
  ];

  const labels = [1,1,1,1,0,0,0,0];  // 1 = Positive, 0 = Negative

  // Convert text → numbers (simple encoding)
  const vocab = {};
  let index = 1;

  sentences.forEach(s => {
    s.toLowerCase().split(" ").forEach(w => {
      if (!vocab[w]) vocab[w] = index++;
    });
  });

  function encode(sentence) {
    const arr = new Array(10).fill(0);
    sentence.toLowerCase().split(" ").forEach((w,i) => {
      arr[i] = vocab[w] || 0;
    });
    return arr;
  }

  const xs = tf.tensor2d(sentences.map(encode));
  const ys = tf.tensor2d(labels, [labels.length,1]);

  // Model
  const model = tf.sequential();
  model.add(tf.layers.dense({units:16, activation:'relu', inputShape:[10]}));
  model.add(tf.layers.dense({units:8, activation:'relu'}));
  model.add(tf.layers.dense({units:1, activation:'sigmoid'}));

  model.compile({
    optimizer: tf.train.adam(0.05),
    loss: 'binaryCrossentropy',
    metrics:['accuracy']
  });

  console.log("Training Model...");
  await model.fit(xs, ys, {epochs:150});
  console.log("Training Completed");

  // ===== TEST CUSTOM SENTENCE =====
  const testSentence = "I love this";
  const testTensor = tf.tensor2d([encode(testSentence)]);
  const pred = model.predict(testTensor);
  const confidence = pred.dataSync()[0];

  console.log("Sentence:", testSentence);
  console.log("Confidence:", confidence.toFixed(3));

  if (confidence > 0.6) {
    console.log("Prediction: POSITIVE 😊");
  } else {
    console.log("Prediction: NEGATIVE 😞");
  }
}

run();
