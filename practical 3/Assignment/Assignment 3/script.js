console.log("Comparing Dense vs RNN...");

const xs = tf.randomNormal([100,5]);
const ys = tf.randomUniform([100,1]);

async function trainDense() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units:8, activation:'relu', inputShape:[5]}));
  model.add(tf.layers.dense({units:1}));

  model.compile({optimizer:'adam', loss:'meanSquaredError'});
  await model.fit(xs, ys, {epochs:5});
  console.log("Dense Training Done");
}

async function trainRNN() {
  const model = tf.sequential();
  model.add(tf.layers.simpleRNN({units:8, inputShape:[5,1]}));
  model.add(tf.layers.dense({units:1}));

  model.compile({optimizer:'adam', loss:'meanSquaredError'});
  await model.fit(xs.reshape([100,5,1]), ys, {epochs:5});
  console.log("RNN Training Done");
}

(async()=>{
  await trainDense();
  await trainRNN();
  console.log("Comparison Completed");
})();
