const imageUpload = document.getElementById("fileUpload");
const image = document.getElementById("inputImage");
const canvas = document.getElementById("overlay");

let faceMatcher;

// Load all required models
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("./models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("./models")
]).then(start);

async function start() {
  const labeledFaceDescriptors = await loadLabeledImages();
  faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  console.log("Models & labeled images loaded");
}

// When user uploads image
imageUpload.addEventListener("change", () => {
  image.src = URL.createObjectURL(imageUpload.files[0]);
});

// Detect & recognize faces
async function detectFaces() {
  if (!faceMatcher) {
    alert("Models are still loading. Please wait...");
    return;
  }

  const detections = await faceapi
    .detectAllFaces(image)
    .withFaceLandmarks()
    .withFaceDescriptors();

  canvas.width = image.width;
  canvas.height = image.height;

  const resized = faceapi.resizeResults(detections, {
    width: image.width,
    height: image.height
  });

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  resized.forEach(detection => {
    const box = detection.detection.box;
    const match = faceMatcher.findBestMatch(detection.descriptor);

    // Draw box
    ctx.strokeStyle = "#00ffcc";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw name
    ctx.fillStyle = "#00ffcc";
    ctx.font = "16px Poppins";
    ctx.fillText(match.label, box.x, box.y - 5);
  });
}

// Load labeled images
function loadLabeledImages() {
  const labels = [
    "Black Widow",
    "Captain America",
    "Captain Marvel",
    "Hawkeye",
    "Jim Rhodes",
    "Thor",
    "Tony Stark"
  ];

  return Promise.all(
    labels.map(async label => {
      const descriptions = [];

      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `./labeled_images/${label}/${i}.jpg`
        );

        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detections) {
          descriptions.push(detections.descriptor);
        }
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
