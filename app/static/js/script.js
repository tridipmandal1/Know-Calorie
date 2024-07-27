document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.querySelector("input[type='file']");
    const customFileUpload = document.querySelector(".custom-file-upload");

    // Make sure the customFileUpload is a label and its 'for' attribute matches the file input's id
    fileInput.id = 'fileInput';
    customFileUpload.setAttribute('for', 'fileInput');

    fileInput.addEventListener("change", function() {
        const fileName = this.value.split("\\").pop();
        if (fileName) {
            customFileUpload.textContent = fileName;
        } else {
            customFileUpload.textContent = "Choose file";
        }
    });
});