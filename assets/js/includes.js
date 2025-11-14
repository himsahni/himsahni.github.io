// Function to load HTML includes
function loadHTML(elementId, file) {
    fetch(file)
        .then(response => response.text())
        .then(data => {
            document.getElementById(elementId).innerHTML = data;
        })
        .catch(error => console.error('Error loading ' + file + ':', error));
}

// Load header and footer when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadHTML('header-placeholder', 'header.html');
    loadHTML('footer-placeholder', 'footer.html');
});