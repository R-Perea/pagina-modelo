let model, webcam, canvas, ctx, labelEl;

async function init() {
  model = await tflite.loadTFLiteModel('assets/vidrio_plastico_basura.tflite');
  labelEl = document.getElementById('label');
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');

  const video = document.getElementById('webcam');
  webcam = await tf.data.webcam(video);
  loop();
}

async function loop() {
  const img = await webcam.capture();
  const resized = tf.image.resizeBilinear(img, [100, 100])
                     .mean(2) // convierte a gris
                     .expandDims(2) // añade canal
                     .expandDims(0) // añade batch
                     .toFloat()
                     .div(255);
  const output = model.predict(resized);
  const scores = output.dataSync(); // array con 3 valores

  const labels = ['glass', 'plastic', 'trash'];
  const idx = scores.indexOf(Math.max(...scores));
  labelEl.innerText = `${labels[idx]} (${(scores[idx]*100).toFixed(2)}%)`;

  // Dibuja el frame en canvas
  canvas.width = img.shape[1];
  canvas.height = img.shape[0];
  ctx.drawImage(webcam.canvas, 0, 0);

  img.dispose();
  resized.dispose();
  output.dispose();

  requestAnimationFrame(loop);
}

init();
