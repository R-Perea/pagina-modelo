(async () => {
  const video = document.getElementById('webcam');
  await tf.ready();
  const model = await tflite.loadTFLiteModel(
    'https://cdn.jsdelivr.net/gh/<usuario>/<repo>@main/assets/vidrio_plastico_basura.tflite'
  );

  const cam = await tf.data.webcam(video);
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const snapshotEl = document.getElementById('snapshot');
  const labelEl = document.getElementById('label');
  const labels = ['glass', 'plastic', 'trash'];

  async function predictFrame() {
    const img = await cam.capture(); // tf.Tensor3D
    const resized = tf.image.resizeBilinear(img, [100, 100])
                      .expandDims(0)
                      .toFloat()
                      .div(255);

    const tfliteOut = await model.predict(resized);
    const confidences = tfliteOut.dataSync();
    const max = confidences.indexOf(Math.max(...confidences));

    labelEl.textContent = `${labels[max]} ${(confidences[max] * 100).toFixed(2)}%`;

    // Dibuja fotograma estilizado en canvas
    const [w, h] = [video.videoWidth, video.videoHeight];
    canvas.width = w; canvas.height = h;
    ctx.drawImage(video, 0, 0, w, h);
    snapshotEl.src = canvas.toDataURL('image/jpeg');

    img.dispose();
    resized.dispose();
    requestAnimationFrame(predictFrame);
  }

  predictFrame();
})();
