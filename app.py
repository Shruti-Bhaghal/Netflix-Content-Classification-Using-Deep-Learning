import os
import re
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Determine frontend directory (check 'frontened', 'frontend', or BASE_DIR)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontened")
if not os.path.exists(FRONTEND_DIR) or not os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if not os.path.exists(FRONTEND_DIR) or not os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    FRONTEND_DIR = BASE_DIR

print(f"Serving frontend from: {FRONTEND_DIR}")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# Load the saved model bundle
MODEL_FILE = os.path.join(BASE_DIR, "netflix_model.pkl")
if not os.path.exists(MODEL_FILE):
    MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

print(f"Loading model bundle from {MODEL_FILE}...")
with open(MODEL_FILE, "rb") as f:
    bundle = pickle.load(f)

movies_df = bundle['movies_df']
tfidf = bundle['tfidf']
tfidf_matrix = bundle['tfidf_matrix']
ann_model = bundle.get('ann_model')
label_encoder = bundle.get('label_encoder')

# Precompute titles mapping for fast case-insensitive lookup
title_map = {}
for idx, title in enumerate(movies_df['title']):
    title_map[str(title).strip().lower()] = idx


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    if os.path.exists(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    return jsonify({"error": "File not found"}), 404


@app.route("/movies", methods=["GET"])
def get_movies():
    """Returns catalog of movie & show titles for autocomplete."""
    titles = movies_df['title'].dropna().tolist()
    return jsonify(titles)


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Given a movie/show title, recommends top similar titles
    using the ANN / TF-IDF deep feature representations.
    """
    data = request.get_json(silent=True) or {}
    movie_name = data.get("movie", "").strip()

    if not movie_name:
        return jsonify({"error": "Please provide a movie name."}), 400

    query_lower = movie_name.lower()

    # Direct match or partial match lookup
    target_idx = title_map.get(query_lower)
    if target_idx is None:
        matches = [idx for t, idx in title_map.items() if query_lower in t]
        if matches:
            target_idx = matches[0]

    if target_idx is None:
        # Not found in catalog: vectorize custom text and match
        custom_vec = tfidf.transform([query_lower])
        sim_scores = cosine_similarity(custom_vec, tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[::-1][:6]
    else:
        # Find similar items using feature vectors
        target_vec = tfidf_matrix[target_idx]
        sim_scores = cosine_similarity(target_vec, tfidf_matrix).flatten()
        sim_scores[target_idx] = -1
        top_indices = sim_scores.argsort()[::-1][:6]

    recommendations = []
    for idx in top_indices:
        row = movies_df.iloc[idx]
        score = float(sim_scores[idx]) if target_idx is not None else float(sim_scores[idx])
        recommendations.append({
            "title": str(row["title"]),
            "type": str(row["type"]),
            "year": str(row["release_year"]),
            "rating": str(row["rating"]),
            "duration": str(row.get("duration", "")),
            "genre": str(row.get("listed_in", "")),
            "description": str(row.get("description", "")),
            "match_score": f"{max(0.0, score):.1%}"
        })

    return jsonify({"recommendations": recommendations})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts Movie vs TV Show using the trained Artificial Neural Network (ANN).
    """
    data = request.get_json(silent=True) or {}
    country = data.get("country", "Unknown")
    release_year = data.get("release_year", 2020)
    rating = data.get("rating", "TV-MA")
    duration = data.get("duration", "90 min")
    genre = data.get("genre", "Drama")
    description = data.get("description", "")

    # Clean genre as trained
    genre_clean = re.sub(r'\bTV\b|\bShows?\b|\bMovies?\b', ' ', str(genre), flags=re.IGNORECASE)
    genre_clean = re.sub(r'\s+', ' ', genre_clean).strip()

    features_text = (
        str(country) + ' ' +
        str(rating) + ' ' +
        str(duration) + ' ' +
        genre_clean + ' ' +
        str(description) + ' ' +
        str(release_year)
    )

    vec = tfidf.transform([features_text])
    pred_idx = ann_model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    probs = ann_model.predict_proba(vec)[0]

    return jsonify({
        "predicted_type": pred_label,
        "confidence": float(max(probs)),
        "probabilities": {cls: float(p) for cls, p in zip(label_encoder.classes_, probs)}
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Backend Server running at http://localhost:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
