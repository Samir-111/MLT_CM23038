console.log("Preparing dataset...");

// Small dataset
const sentences = [
  "I love this product",
  "This is amazing",
  "Very happy with service",
  "Best experience ever",
  "I hate this",
  "Very bad product",
  "Worst service",
  "Not good at all"
];

const labels = [1,1,1,1,0,0,0,0]; // 1 = Positive, 0 = Negative

// Simple tokenizer
const tokenizer = {};
let index = 1;

sentences.forEach(sentence => {
  sentence.toLowerCase().split(" ").forEach(word => {
    if (!tokenizer[word]) tokenizer[word] = index++;
  });
});

// Convert text → numbers
function encode(sentence) {
  return sentence.toLowerCase().split(" ").map(w => tokenizer[w] || 0);
}

// Pad sequences
function pad(seq, maxLen=5) {
  while (seq.length < maxLen) seq.push(0);
  return seq.slice(0, maxLen);
}

const xs = tf.tensor2d(sentences.map(s => pad(encode(s))));
const ys = tf.tensor2d(labels, [labels.length, 1]);

// =====================
// DENSE MODEL
// =====================
async function trainDense() {
  console.log("Training Dense Model...");

  const model = tf.sequential();
  model.add(tf.layers.embedding({inputDim: 50, outputDim: 8, inputLength: 5}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 6, activation: "relu"}));
  model.add(tf.layers.dense({units: 1, activation: "sigmoid"}));

  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  await model.fit(xs, ys, {epochs: 20});
  return model;
}

// =====================
// RNN MODEL
// =====================
async function trainRNN() {
  console.log("Training RNN Model...");

  const model = tf.sequential();
  model.add(tf.layers.embedding({inputDim: 50, outputDim: 8, inputLength: 5}));
  model.add(tf.layers.simpleRNN({units: 8}));
  model.add(tf.layers.dense({units: 1, activation: "sigmoid"}));

  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  await model.fit(xs, ys, {epochs: 20});
  return model;
}

// =====================
// TEST FUNCTION
// =====================
function predict(model, text) {
  const input = tf.tensor2d([pad(encode(text))]);
  const output = model.predict(input);
  const score = output.dataSync()[0];

  console.log(`Sentence: "${text}"`);
  console.log(`Confidence: ${(score*100).toFixed(2)}%`);
  console.log(score > 0.5 ? "Prediction: Positive 😊" : "Prediction: Negative 😡");
}

// =====================
// MAIN RUN
// =====================
(async () => {
  const denseModel = await trainDense();
  const rnnModel = await trainRNN();

  console.log("------ TESTING ------");

  predict(denseModel, "I love this");
  predict(rnnModel, "I love this");

  predict(denseModel, "Worst product");
  predict(rnnModel, "Worst product");

  console.log("Dense vs RNN comparison done");
})();
