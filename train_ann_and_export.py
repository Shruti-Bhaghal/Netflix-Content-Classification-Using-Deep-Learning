"""
train_ann_and_export.py
Trains the ANN model and recommendation system based on deeplearningcu (1).ipynb,
and exports all requested .pkl files.
"""

import os
import re
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

print("1. Loading dataset...")
csv_path = "netflix_titles.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found.")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} titles.")

# Preprocessing matching the notebook
text_columns = ['country', 'rating', 'duration', 'listed_in', 'description']
for col in text_columns:
    df[col] = df[col].fillna('Unknown').astype(str)

if 'release_year' in df.columns:
    df['release_year'] = df['release_year'].fillna(2020).astype(int)
else:
    df['release_year'] = 2020

# Clean genre as in Notebook Cell 11
df['genre_clean'] = (
    df['listed_in']
    .str.replace(r'\bTV\b|\bShows?\b|\bMovies?\b', ' ', regex=True, case=False)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

# Combined features as in Notebook Cell 11 & 43
df['features'] = (
    df['country'].astype(str) + ' ' +
    df['rating'].astype(str) + ' ' +
    df['duration'].astype(str) + ' ' +
    df['genre_clean'] + ' ' +
    df['description'].astype(str) + ' ' +
    df['release_year'].astype(str)
)

# Filter valid types
valid_df = df[df['type'].isin(['Movie', 'TV Show'])].copy().reset_index(drop=True)

# Encode target
le = LabelEncoder()
y = le.fit_transform(valid_df['type'])

# TF-IDF Vectorizer (5000 max features, matching Notebook Cell 19)
print("2. Fitting TF-IDF Vectorizer (5000 features)...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(valid_df['features'])

# Train/Test Split matching Notebook Cell 17
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training records: {X_train.shape[0]}, Testing records: {X_test.shape[0]}")

# Train ANN (Multi-Layer Perceptron matching Notebook Cell 21: Dense 128 -> Dense 64 -> Dense 32)
print("3. Training Artificial Neural Network (ANN: 128 -> 64 -> 32)...")
ann_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=30,
    random_state=42,
    verbose=True
)
ann_model.fit(X_train, y_train)

# Evaluate
y_pred = ann_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"ANN Model Test Accuracy: {acc:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 4. Build Recommendation Engine Matrix
print("4. Computing Feature Embeddings for Movie Recommender...")
# Compute similarity or store TF-IDF matrix for all items
# Also extract ANN latent layer representations if desired
movies_clean = valid_df[['show_id', 'type', 'title', 'director', 'country', 'release_year', 'rating', 'duration', 'listed_in', 'description']].copy()
# Normalized title for case-insensitive lookup
movies_clean['title_lower'] = movies_clean['title'].str.strip().str.lower()

# Export comprehensive model bundle
recommender_bundle = {
    'movies_df': movies_clean,
    'tfidf': tfidf,
    'tfidf_matrix': X,
    'ann_model': ann_model,
    'label_encoder': le,
    'classes': list(le.classes_)
}

# 5. Export .pkl files
print("5. Exporting .pkl files...")

# Main model.pkl bundle
with open('model.pkl', 'wb') as f:
    pickle.dump(recommender_bundle, f)
print("Saved: model.pkl")

with open('netflix_model.pkl', 'wb') as f:
    pickle.dump(recommender_bundle, f)
print("Saved: netflix_model.pkl")

# Standalone ANN model .pkl
joblib.dump(ann_model, 'netflix_ann_model.pkl')
print("Saved: netflix_ann_model.pkl")

# Preprocessing objects matching Notebook Cell 45
joblib.dump(tfidf, 'netflix_tfidf_vectorizer.pkl')
print("Saved: netflix_tfidf_vectorizer.pkl")

joblib.dump(le, 'netflix_label_encoder.pkl')
print("Saved: netflix_label_encoder.pkl")

# Recommender bundle
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump({
        'movies_df': movies_clean,
        'tfidf': tfidf,
        'tfidf_matrix': X
    }, f)
print("Saved: recommendation_model.pkl")

print("\n--- ALL .PKL FILES CREATED SUCCESSFULLY ---")

