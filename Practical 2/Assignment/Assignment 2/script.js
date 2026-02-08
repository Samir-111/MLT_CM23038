let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;

  ctx.fillStyle = "black";
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 8, 0, Math.PI * 2);
  ctx.fill();
}

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "";
}

async function predictDigit() {

  // Convert canvas → image tensor
  let imgData = ctx.getImageData(0, 0, 280, 280);
  let tensor = tf.browser.fromPixels(imgData, 1);

  tensor = tf.image.resizeBilinear(tensor, [28, 28]);
  tensor = tensor.toFloat().div(255.0);
  tensor = tensor.reshape([1, 28, 28, 1]);

  // Simple trained-like model (for demo)
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  const prediction = model.predict(tensor);
  const digit = prediction.argMax(-1).dataSync()[0];

  document.getElementById("result").innerText =
    "Predicted Digit: " + digit;
}
