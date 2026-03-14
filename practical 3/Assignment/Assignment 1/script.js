console.log("Training Sentiment Model...");

// Same dataset as practical
const xs = tf.tensor2d([
  [1,2,3,0,0],
  [4,5,6,0,0],
  [7,8,9,0,0],
  [10,11,12,0,0]
]);

const ys = tf.tensor2d([[1],[1],[0],[0]]);

const model = tf.sequential();
model.add(tf.layers.dense({units: 8, activation:'relu', inputShape:[5]}));
model.add(tf.layers.dense({units: 1, activation:'sigmoid'}));

model.compile({
  optimizer:'adam',
  loss:'binaryCrossentropy',
  metrics:['accuracy']
});

(async () => {
  await model.fit(xs, ys, {epochs:15});
  console.log("Training Completed");
})();

