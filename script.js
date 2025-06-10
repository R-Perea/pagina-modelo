const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// 1) Carga del modelo
const model = await tflite.loadTFLiteModel('assets/vidrio_plastico_basura.tflite');


// 2) Activa webcam
const webcam = await tf.data.webcam(video);

// 3) Ciclo de inferencia
async function predictLoop() {
  const img = await webcam.capture();
  const input = tf.browser
    .fromPixels(img)
    .resizeBilinear([100, 100])
    .mean(2)        // convierte a gris
    .expandDims(2)  // añade canal
    .expandDims(0)  // añade batch
    .toFloat()
    .div(255);

  const output = model.predict(input);
  const confidences = await output.data();
  console.log('Confidences:', confidences);

  img.dispose();
  input.dispose();
  output.dispose();

  requestAnimationFrame(predictLoop);
}

predictLoop();
