let model, webcam, canvas, ctx, labelEl;

async function init() {
  model = await tflite.loadTFLiteModel('assets/vidrio_plastico_basura.tflite');

  const video = document.getElementById('webcam');

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        resolve();
      };
    });
    webcam = await tf.data.webcam(video); // Esta línea es suficiente
 

  labelEl = document.getElementById('label');
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');

  requestAnimationFrame(loop);
}

async function loop() {
  const img = await webcam.capture();

  const resized = tf.tidy(() => img
    .resizeBilinear([100, 100])
    .mean(2)
    .expandDims(2)
    .expandDims(0)
    .toFloat()
    .div(255)
  );

  const output = model.predict(resized);
  const scores = await output.data();

  const labels = ['glass', 'plastic', 'trash'];
  const idx = scores.indexOf(Math.max(...scores));
  labelEl.innerText = `${labels[idx]} (${(scores[idx] * 100).toFixed(2)}%)`;

  canvas.width = img.shape[1];
  canvas.height = img.shape[0];
  ctx.drawImage(webcam.canvas, 0, 0);

  img.dispose();
  resized.dispose();
  output.dispose();

  requestAnimationFrame(loop);
}

init();
window.init = init;
document.getElementById('startBtn').addEventListener('click', init);
