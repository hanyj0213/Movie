import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Recommender", layout="wide")


# -----------------------------
# Data loading (cached)
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")   # movieId, title, genres
    ratings = pd.read_csv("ratings.csv") # userId, movieId, rating, timestamp
    links = pd.read_csv("links.csv")     # movieId, imdbId, tmdbId
    tags = pd.read_csv("tags.csv")       # userId, movieId, tag, timestamp
    return movies, ratings, links, tags


@st.cache_data
def build_top_tags(tags: pd.DataFrame, top_n: int = 5):
    # Count tags per movie and keep top N
    tmp = (
        tags.groupby(["movieId", "tag"])
            .size()
            .reset_index(name="count")
            .sort_values(["movieId", "count"], ascending=[True, False])
    )
    top = (
        tmp.groupby("movieId")
           .head(top_n)
           .groupby("movieId")["tag"]
           .apply(list)
           .to_dict()
    )
    return top


@st.cache_data
def build_mappings(movies: pd.DataFrame, links: pd.DataFrame):
    movie_title = dict(zip(movies["movieId"], movies["title"]))
    movie_genres = dict(zip(movies["movieId"], movies["genres"]))

    links_map = links.set_index("movieId")[["imdbId", "tmdbId"]].to_dict("index")

    def imdb_url(movie_id: int):
        info = links_map.get(movie_id)
        if not info or pd.isna(info.get("imdbId")):
            return None
        return f"https://www.imdb.com/title/tt{int(info['imdbId']):07d}/"

    return movie_title, movie_genres, imdb_url


# -----------------------------
# Model building (cached)
# -----------------------------
@st.cache_resource
def train_item_knn(ratings: pd.DataFrame, k_neighbors: int = 50):
    """
    Train an item-item KNN model (cosine similarity) on a user-item sparse matrix.
    Returns: (knn_model, item_matrix, mappings)
    """
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_id_to_idx = {m: i for i, m in enumerate(movie_ids)}
    idx_to_movie_id = {i: m for m, i in movie_id_to_idx.items()}

    rows = ratings["userId"].map(user_id_to_idx).values
    cols = ratings["movieId"].map(movie_id_to_idx).values
    data = ratings["rating"].values.astype(np.float32)

    R = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    item_matrix = R.T  # items x users

    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_neighbors + 1)
    knn.fit(item_matrix)

    mappings = {
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_id_to_idx": user_id_to_idx,
        "movie_id_to_idx": movie_id_to_idx,
        "idx_to_movie_id": idx_to_movie_id,
        "R": R
    }
    return knn, item_matrix, mappings


def similar_movies(movie_id: int, knn, item_matrix, mappings, movie_title, movie_genres, imdb_url, top_tags_dict, n: int = 10):
    movie_id_to_idx = mappings["movie_id_to_idx"]
    idx_to_movie_id = mappings["idx_to_movie_id"]

    if movie_id not in movie_id_to_idx:
        raise ValueError("movieId not found in ratings data (maybe filtered out).")

    item_idx = movie_id_to_idx[movie_id]
    distances, indices = knn.kneighbors(item_matrix[item_idx], n_neighbors=n + 1)

    sims = 1.0 - distances.flatten()
    neigh = indices.flatten()

    out = []
    for sim, j in zip(sims[1:], neigh[1:]):  # skip itself
        mid = int(idx_to_movie_id[j])
        out.append({
            "movieId": mid,
            "title": movie_title.get(mid, str(mid)),
            "genres": movie_genres.get(mid, ""),
            "similarity": float(sim),
            "top_tags": ", ".join(top_tags_dict.get(mid, [])[:5]),
            "imdb": imdb_url(mid)
        })

    return pd.DataFrame(out)


def recommend_for_user(user_id: int, knn, item_matrix, mappings, movie_title, movie_genres, imdb_url, top_tags_dict,
                       n_recs: int = 10, min_rating: float = 4.0, k_neighbors: int = 50):
    """
    Personalized recommendations:
      - take user's liked movies (rating >= min_rating)
      - for each liked movie, find similar movies via item-item KNN
      - score candidates by (similarity * user's rating), then sum
    """
    user_id_to_idx = mappings["user_id_to_idx"]
    idx_to_movie_id = mappings["idx_to_movie_id"]
    R = mappings["R"]

    if user_id not in user_id_to_idx:
        raise ValueError("Unknown userId.")

    uidx = user_id_to_idx[user_id]
    user_row = R[uidx]

    if user_row.nnz == 0:
        raise ValueError("This user has no ratings.")

    seen_items = set(user_row.indices)

    liked_mask = user_row.data >= min_rating
    liked_items = user_row.indices[liked_mask]
    liked_ratings = user_row.data[liked_mask]

    # fallback: if no liked items, use user's top rated items
    if len(liked_items) == 0:
        order = np.argsort(-user_row.data)
        liked_items = user_row.indices[order[:5]]
        liked_ratings = user_row.data[order[:5]]

    scores = {}

    for item_idx, r in zip(liked_items, liked_ratings):
        distances, indices = knn.kneighbors(item_matrix[item_idx], n_neighbors=k_neighbors + 1)
        sims = 1.0 - distances.flatten()
        neigh = indices.flatten()

        for sim, j in zip(sims[1:], neigh[1:]):
            if j in seen_items:
                continue
            scores[j] = scores.get(j, 0.0) + float(sim * r)

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]

    out = []
    for item_idx, score in top:
        mid = int(idx_to_movie_id[item_idx])
        out.append({
            "movieId": mid,
            "title": movie_title.get(mid, str(mid)),
            "genres": movie_genres.get(mid, ""),
            "score": float(score),
            "top_tags": ", ".join(top_tags_dict.get(mid, [])[:5]),
            "imdb": imdb_url(mid)
        })

    return pd.DataFrame(out)


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommender (Item-Item KNN CF)")

movies, ratings, links, tags = load_data()
top_tags_dict = build_top_tags(tags, top_n=5)
movie_title, movie_genres, imdb_url = build_mappings(movies, links)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Similar Movies", "Personalized (userId)"])
    k_neighbors = st.slider("K neighbors", 10, 200, 50, 10)
    n_results = st.slider("Number of results", 5, 30, 10, 1)
    min_rating = st.slider("Min rating for 'liked' movies", 3.0, 5.0, 4.0, 0.5)

knn, item_matrix, mappings = train_item_knn(ratings, k_neighbors=k_neighbors)

if mode == "Similar Movies":
    st.subheader("Find movies similar to a selected movie")

    query = st.text_input("Search movie title")
    if query.strip():
        candidates = movies[movies["title"].str.lower().str.contains(query.lower(), na=False)].head(50)
    else:
        candidates = movies.head(50)

    selected = st.selectbox(
        "Pick a movie",
        options=candidates["movieId"].tolist(),
        format_func=lambda mid: movie_title.get(mid, str(mid))
    )

    if st.button("Recommend similar movies"):
        with st.spinner("Computing similarities..."):
            df = similar_movies(
                movie_id=int(selected),
                knn=knn,
                item_matrix=item_matrix,
                mappings=mappings,
                movie_title=movie_title,
                movie_genres=movie_genres,
                imdb_url=imdb_url,
                top_tags_dict=top_tags_dict,
                n=n_results
            )
        st.dataframe(df, use_container_width=True)

else:
    st.subheader("Personalized recommendations by userId")

    user_ids = sorted(ratings["userId"].unique().tolist())
    user_id = st.selectbox("Select a userId", options=user_ids)

    if st.button("Get personalized recommendations"):
        with st.spinner("Generating recommendations..."):
            df = recommend_for_user(
                user_id=int(user_id),
                knn=knn,
                item_matrix=item_matrix,
                mappings=mappings,
                movie_title=movie_title,
                movie_genres=movie_genres,
                imdb_url=imdb_url,
                top_tags_dict=top_tags_dict,
                n_recs=n_results,
                min_rating=min_rating,
                k_neighbors=k_neighbors
            )
        st.dataframe(df, use_container_width=True)
