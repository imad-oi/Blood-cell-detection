let imageCountPerRow = 0;
const images = [];


function search() {
  alert("Search functionality is not yet implemented.");
}

let currentImageName = "";
const predictions = [];

function uploadImage(event) {
  const imgContainer = document.querySelector(".images");
  const file = event.target.files[0];
  currentImageName = file.name;
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const img = document.createElement("img");
      img.src = e.target.result;
      img.alt = "Uploaded Image";
      imgContainer.appendChild(img);
    };
    reader.readAsDataURL(file);

    // Send the image to Flask server for processing
    const formData = new FormData();
    formData.append("image", file);

    fetch("http://127.0.0.1:5000", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        predictions.push(data.predictions);
        imageCountPerRow += 1;
      })
      .catch((error) => console.error("Error:", error));
  }
}

const imgContainer = document.querySelector(".images");

async function extractChannelA() {
  const img = document.createElement("img");

  img.src = `static/images/a_channel_${currentImageName}`
  img.alt = currentImageName;
  img.onerror = () => console.error(`Failed to load image a_channel_${currentImageName}`);
  img.onload = () => console.log(`Loaded image a_channel_${currentImageName}`);
  imgContainer.appendChild(img);
  imageCountPerRow += 1;
  
}

function applyThreshold() {
  const img = document.createElement("img");
  img.src = `static/images/thresholded_${currentImageName}`
  img.alt = currentImageName;
  img.onerror = () => console.error(`Failed to load image thresholded_${currentImageName}`);
  img.onload = () => console.log(`Loaded image thresholded_${currentImageName}`);
  imgContainer.appendChild(img);
  imageCountPerRow += 1;

}

function segmentImage() {
  const img = document.createElement("img");
  img.src = `static/images/segmented_${currentImageName}`
  img.alt = currentImageName;
  img.onerror = () => console.error(`Failed to load image segmented_${currentImageName}`);
  img.onload = () => console.log(`Loaded image segmented_${currentImageName}`);
  imgContainer.appendChild(img);
  imageCountPerRow += 1;
}

function applyWatershed() {
  const img = document.createElement("img");
  img.src = `static/images/watershed_${currentImageName}`
  img.alt = currentImageName;
  img.onerror = () => console.error(`Failed to load image watershed_${currentImageName}`);
  img.onload = () => console.log(`Loaded image watershed_${currentImageName}`);
  imgContainer.appendChild(img);
  imageCountPerRow += 1;
}

function predictClass() {
  // Placeholder for actual prediction logic
  console.log(predictions);
  const predictedClass = predictions[0].svm;
  document.getElementById("predictedClass").innerText = predictedClass;
}




// function uploadImage(event) {
//     const imgContainer = document.querySelector('.images');
//     const file = event.target.files[0];
//     if (file) {
//         const reader = new FileReader();
//         reader.onload = function(e) {
//             const img = document.createElement('img');
//             img.src = e.target.result;
//             img.alt = 'Uploaded Image';
//             imgContainer.appendChild(img);
//         };
//         reader.readAsDataURL(file);
//     }
// }