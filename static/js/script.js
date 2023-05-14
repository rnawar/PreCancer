let menu = document.querySelector("#menu-icon");
let navbar = document.querySelector(".navbar");
let title = document.getElementById("title");

menu.onclick = () => {
  menu.classList.toggle("bx-x");
  navbar.classList.toggle("open");
};

window.addEventListener("scroll", () => {
  let value = window.scrollY;

  title.style.marginTop = value * 2.75 + "px";
});

document.querySelectorAll(".input-file").forEach((inputElement) => {
  const dropZoneElement = inputElement.closest(".drop-zone");

  dropZoneElement.addEventListener("click", (e) => {
    inputElement.click();
  });

  inputElement.addEventListener("change", (e) => {
    if (inputElement.files.length) {
      updateThumbnail(dropZoneElement, inputElement.files[0]);
    }
  });

  dropZoneElement.addEventListener("dragover", (e) => {
    e.preventDefault();
  });
  dropZoneElement.addEventListener("drop", (e) => {
    e.preventDefault();

    if (e.dataTransfer.files.length) {
      inputElement.files = e.dataTransfer.files;
      updateThumbnail(dropZoneElement, e.dataTransfer.files[0]);
    }
  });
});

function updateThumbnail(dropZoneElement, file) {
  let thumbnailElement = dropZoneElement.querySelector(".drop-zone-thumb");

  if (dropZoneElement.querySelector(".drop-zone-prompt")) {
    dropZoneElement.querySelector(".drop-zone-prompt").remove();
    dropZoneElement.querySelector("#upload-icon").remove();
  }

  if (!thumbnailElement) {
    thumbnailElement = document.createElement("div");
    thumbnailElement.classList.add("drop-zone-thumb");
    dropZoneElement.appendChild(thumbnailElement);
  }

  thumbnailElement.dataset.label = file.name;
}
