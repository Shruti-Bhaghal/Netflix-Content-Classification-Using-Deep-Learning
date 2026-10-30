// ====== CONFIG ======
// Change this to your deployed backend URL (e.g. "https://your-app.onrender.com").
// Leave as "" if the backend serves this frontend from the same domain.
const API_BASE_URL = "";

const RECOMMEND_ENDPOINT = `${API_BASE_URL}/recommend`; // POST {movie: "..."}
const MOVIES_ENDPOINT = `${API_BASE_URL}/movies`;       // GET -> ["Avatar", "Titanic", ...] (optional)

// ====== ELEMENTS ======
const form = document.getElementById("recommend-form");
const input = document.getElementById("movie-input");
const button = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const datalist = document.getElementById("movie-list");

// ====== HELPERS ======
function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

// Accepts either a string ("Avatar") or an object ({title, poster, year, rating})
function normalizeMovie(item) {
  if (typeof item === "string") return { title: item };
  return {
    title: item.title || item.name || "Unknown",
    poster: item.poster || item.poster_url || null,
    year: item.year || item.release_date?.slice(0, 4) || "",
    rating: item.rating || item.vote_average || "",
  };
}

function createCard(movie) {
  const card = document.createElement("article");
  card.className = "card";

  let posterEl;
  if (movie.poster) {
    posterEl = document.createElement("img");
    posterEl.src = movie.poster;
    posterEl.alt = movie.title;
    posterEl.loading = "lazy";
    posterEl.onerror = () => posterEl.replaceWith(createPlaceholder());
  } else {
    posterEl = createPlaceholder();
  }

  const body = document.createElement("div");
  body.className = "card-body";

  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = movie.title;
  body.appendChild(title);

  const metaParts = [movie.year, movie.rating ? `⭐ ${movie.rating}` : ""].filter(Boolean);
  if (metaParts.length) {
    const meta = document.createElement("div");
    meta.className = "card-meta";
    meta.textContent = metaParts.join(" • ");
    body.appendChild(meta);
  }

  card.append(posterEl, body);
  return card;
}

function createPlaceholder() {
  const div = document.createElement("div");
  div.className = "poster-placeholder";
  div.textContent = "🎞️";
  return div;
}

// ====== LOAD MOVIE LIST FOR AUTOCOMPLETE (optional) ======
async function loadMovieList() {
  try {
    const res = await fetch(MOVIES_ENDPOINT);
    if (!res.ok) return;
    const movies = await res.json();
    const fragment = document.createDocumentFragment();
    movies.slice(0, 5000).forEach((name) => {
      const option = document.createElement("option");
      option.value = typeof name === "string" ? name : name.title;
      fragment.appendChild(option);
    });
    datalist.appendChild(fragment);
  } catch (err) {
    // Autocomplete is optional; ignore failures.
    console.warn("Could not load movie list:", err);
  }
}

// ====== GET RECOMMENDATIONS ======
async function getRecommendations(movieName) {
  setStatus("Finding recommendations...");
  resultsEl.innerHTML = "";
  button.disabled = true;

  try {
    const res = await fetch(RECOMMEND_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movie: movieName }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || `Server error (${res.status})`);
    }

    const list = data.recommendations || data;
    if (!Array.isArray(list) || list.length === 0) {
      setStatus("No recommendations found. Try another movie.", true);
      return;
    }

    setStatus(`Because you liked "${movieName}":`);
    list.map(normalizeMovie).forEach((m) => resultsEl.appendChild(createCard(m)));
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Something went wrong. Please try again.", true);
  } finally {
    button.disabled = false;
  }
}

// ====== EVENTS ======
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const name = input.value.trim();
  if (name) getRecommendations(name);
});

loadMovieList();
