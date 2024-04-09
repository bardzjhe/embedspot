document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('user-features-form');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const gender = document.getElementById('gender').value;
        const age = document.getElementById('age').value;
        const occupation = document.getElementById('occupation').value;
        const zipCode = document.getElementById('zip-code').value;

        // Here you would typically send a request to your backend to get recommendations
        // For example: axios.post('/api/recommendations', { gender, age, occupation, zipCode }).then(...)
        // This is a mock function to simulate getting recommendations
        getMockRecommendations({ gender, age, occupation, zipCode });
    });
});

function getMockRecommendations(userPreferences) {
    // Simulate a delay and response with some mock recommendations
    setTimeout(() => {
        const recommendations = [
            { title: 'The Shawshank Redemption', year: 1994 },
            { title: 'The Godfather', year: 1972 },
            // Add more mock movie recommendations
        ];

        const resultsElement = document.getElementById('recommendation-results');
        resultsElement.innerHTML = '<h2>Recommended Movies</h2>';
        const list = document.createElement('ul');
        recommendations.forEach(movie => {
            const item = document.createElement('li');
            item.textContent = `${movie.title} (${movie.year})`;
            list.appendChild(item);
        });
        resultsElement.appendChild(list);
    }, 1000);
}