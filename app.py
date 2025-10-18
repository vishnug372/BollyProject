from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model_files/feature_matrix.pkl", "rb") as f:
    feature_matrix = pickle.load(f)

metadata = pd.read_csv("model_files/movie_metadata.csv")

user_likes = []

@app.post("/like/{movie_title}")
def like_movie(movie_title: str):
    if movie_title not in user_likes:
        user_likes.append(movie_title)
    return {"message": f"Added {movie_title} to liked list", "liked_movies": user_likes}

from sklearn.metrics.pairwise import cosine_similarity

@app.get("/next")
def get_next_movie():
    if len(user_likes) == 0:
        # If no likes yet, recommend a random movie
        next_movie = np.random.choice(metadata["title"])
        return {"next_movie": next_movie}

    # Find embeddings for liked movies
    liked_indices = metadata[metadata["title"].isin(user_likes)].index
    liked_vectors = feature_matrix[liked_indices]

    # Compute user preference vector (mean of liked)
    user_vector = np.mean(liked_vectors, axis=0).reshape(1, -1)

    # Compute similarity with all movies
    sims = cosine_similarity(user_vector, feature_matrix).flatten()

    # Avoid recommending already liked ones
    for idx in liked_indices:
        sims[idx] = -1

    # Get the index of the next highest similarity
    next_idx = np.argmax(sims)
    next_movie = metadata.iloc[next_idx]["title"]

    return {"next_movie": next_movie}


@app.post("/reset")
def reset_likes():
    user_likes.clear()
    return {"message": "User likes cleared"}