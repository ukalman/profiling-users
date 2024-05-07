function loadAndDrawChart() {
    const tweetText = document.getElementById('tweetInput').value;
    fetch('http://127.0.0.1:8001/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({tweet: tweetText})
    })
    .then(response => response.json())
    .then(data => {
        const ctx = document.getElementById('raceChart').getContext('2d');
        const labels = Object.keys(data[0]);
        const probabilities = Object.values(data[0]);

        // Clear previous chart if exists
        if (window.bar != undefined) {
            window.bar.destroy();
        }

        window.bar = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Race Prediction Probabilities',
                    data: probabilities,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                },

            }
        });
    })
    .catch(error => console.error('Error fetching data:', error));
}
