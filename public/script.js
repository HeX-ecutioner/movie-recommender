function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    html.setAttribute('data-theme', isDark ? 'light' : 'dark');
}

function renderStars(rating) {
    const full = Math.floor(rating);
    const half = rating - full >= 0.5 ? 1 : 0;
    const empty = 5 - full - half;
    return '★'.repeat(full) + (half ? '⯨' : '') + '☆'.repeat(empty);
}

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('valTotalMovies').innerText = data.total_movies;
            document.getElementById('valTotalRatings').innerText = data.total_ratings;
            document.getElementById('valUniqueUsers').innerText = data.unique_users;


            const tbody = document.getElementById('top5TableBody');
            tbody.innerHTML = '';
            data.top5_movies.forEach(movie => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                            <td>${movie.title}</td>
                            <td>${movie.num_ratings}</td>
                            <td class="stars">${renderStars(movie.avg_rating)}</td>
                        `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Failed to load stats", err);
    }
}

async function searchMovies() {
    const movieInput = document.getElementById('movieInput').value.trim() || 'Batman Begins';
    const contentWeight = document.getElementById('weightInput').value;
    const minRating = document.getElementById('ratingInput').value;

    const grid = document.getElementById('movieGrid');
    const detailsBody = document.getElementById('detailsTableBody');
    const wrapper = document.getElementById('resultsWrapper');
    const loading = document.getElementById('loading');
    const errDiv = document.getElementById('errorMessage');

    wrapper.style.display = 'none';
    errDiv.className = 'error-banner';
    errDiv.innerHTML = '';
    errDiv.style.display = 'none';
    loading.style.display = 'block';

    try {
        const response = await fetch(`/api/recommend?movie=${encodeURIComponent(movieInput)}&content_weight=${contentWeight}&min_rating=${minRating}&top_n=10`);
        if (!response.ok) throw new Error(await response.text() || 'Failed to fetch');

        const data = await response.json();
        loading.style.display = 'none';

        if (data.status === 'ambiguous') {
            errDiv.className = 'ambiguity-banner';
            errDiv.innerHTML = '';

            const contentDiv = document.createElement('div');
            contentDiv.className = 'ambiguity-content';

            const iconDiv = document.createElement('div');
            iconDiv.className = 'ambiguity-icon';
            iconDiv.innerText = '💡';

            const bodyDiv = document.createElement('div');
            bodyDiv.className = 'ambiguity-body';

            const msgP = document.createElement('p');
            msgP.className = 'ambiguity-message';
            msgP.innerText = data.message || `Multiple movies matched "${movieInput}". Please include a release year.`;
            bodyDiv.appendChild(msgP);

            if (data.matches && data.matches.length > 0) {
                const suggDiv = document.createElement('div');
                suggDiv.className = 'ambiguity-suggestions';

                const labelSpan = document.createElement('span');
                labelSpan.className = 'suggestion-label';
                labelSpan.innerText = 'Select a specific title:';
                suggDiv.appendChild(labelSpan);

                const chipsDiv = document.createElement('div');
                chipsDiv.className = 'suggestion-chips';

                data.matches.forEach(m => {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'suggestion-chip';
                    btn.innerText = m.title;
                    btn.title = `Search for ${m.title}`;
                    btn.addEventListener('click', () => {
                        document.getElementById('movieInput').value = m.title;
                        searchMovies();
                    });
                    chipsDiv.appendChild(btn);
                });

                suggDiv.appendChild(chipsDiv);
                bodyDiv.appendChild(suggDiv);
            }

            contentDiv.appendChild(iconDiv);
            contentDiv.appendChild(bodyDiv);
            errDiv.appendChild(contentDiv);
            errDiv.style.display = 'block';

        } else if (data.recommendations && data.recommendations.length > 0) {
            document.getElementById('resultsTitle').innerText = `Top 5 recommendations for ${movieInput}`;
            document.getElementById('resultsTitle').style.display = 'block';

            // Render Top 5 grid
            grid.innerHTML = '';
            data.recommendations.slice(0, 5).forEach((movie, index) => {
                const card = document.createElement('div');
                card.className = 'movie-card';
                card.style.animationDelay = `${index * 0.1}s`;

                const poster = movie.poster_url
                    ? `<img src="${movie.poster_url}" class="movie-poster">`
                    : `<div class="no-poster">No Image</div>`;

                card.innerHTML = `
                            ${poster}
                            <div class="movie-info">
                                <div class="movie-title">${movie.title}</div>
                                <div class="movie-meta">${movie.genres.join(' | ')}</div>
                                <div class="stars">★★★★☆</div>
                                <div class="match-text">Match: ${(movie.score * 100).toFixed(1)}%</div>
                            </div>
                        `;
                grid.appendChild(card);
            });

            // Render Top 10 table
            detailsBody.innerHTML = '';
            data.recommendations.forEach((movie, index) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                            <td>${index + 1}</td>
                            <td>${movie.title}</td>
                            <td>${movie.genres.join(' | ')}</td>
                            <td>${(movie.score * 100).toFixed(2)}%</td>
                        `;
                detailsBody.appendChild(tr);
            });

            wrapper.style.display = 'block';
            document.getElementById('detailedView').style.display = 'block';

        } else {
            errDiv.className = 'error-banner';
            errDiv.innerText = 'No recommendations found.';
            errDiv.style.display = 'block';
        }

    } catch (err) {
        loading.style.display = 'none';
        errDiv.className = 'error-banner';
        errDiv.innerText = err.message;
        errDiv.style.display = 'block';
    }
}

window.addEventListener('DOMContentLoaded', loadStats); // Run stats on load