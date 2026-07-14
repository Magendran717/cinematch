import streamlit as st
import pandas as pd

st.title("🎬 CineMatch")
st.write("An AI-powered movie recommendation engine")

@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

st.subheader("Raw Ratings Data")
st.dataframe(ratings.head())

st.subheader("Raw Movies Data")
st.dataframe(movies.head())

import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Rating Distribution")
fig, ax = plt.subplots()
sns.countplot(x="rating", data=ratings, color="steelblue", ax=ax)
st.pyplot(fig)

st.subheader("Long-Tail: Ratings per Movie")
pop = ratings.groupby("movieId").size().sort_values(ascending=False).reset_index(name="n_ratings")

fig2, ax2 = plt.subplots()
ax2.plot(range(len(pop)), pop["n_ratings"].values, color="firebrick")
ax2.set_xlabel("Movie rank (most → least popular)")
ax2.set_ylabel("# Ratings")
st.pyplot(fig2)

top5_share = pop.head(5)["n_ratings"].sum() / len(ratings)
st.write(f"Top 5 most-rated movies account for **{top5_share:.2%}** of all ratings")

st.subheader("Utility Matrix")
utility = ratings.pivot_table(index="userId", columns="movieId", values="rating")

n_users, n_movies = utility.shape
sparsity = 1 - (len(ratings) / (n_users * n_movies))

st.write(f"Matrix shape: **{n_users} users × {n_movies} movies**")
st.write(f"Sparsity ratio: **{sparsity:.4%}**")
st.dataframe(utility.iloc[:8, :8])

import numpy as np

user_ids = utility.index.values
movie_ids = utility.columns.values
uid_to_idx = {u: i for i, u in enumerate(user_ids)}
mid_to_idx = {m: i for i, m in enumerate(movie_ids)}

matrix = utility.fillna(0).to_numpy(copy=True)

def cosine_sim_matrix(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9   # avoid divide-by-zero for users/movies with no ratings
    Mn = M / norms
    return Mn @ Mn.T

user_sim = cosine_sim_matrix(matrix)     # how similar each user is to every other user
item_sim = cosine_sim_matrix(matrix.T)   # how similar each movie is to every other movie

st.subheader("Similarity Matrices")
st.write(f"User-User similarity matrix shape: {user_sim.shape}")
st.write(f"Item-Item similarity matrix shape: {item_sim.shape}")

def predict_user_based(u_idx, m_idx, k=20):
    sims = user_sim[u_idx].copy()
    sims[u_idx] = 0                          # exclude the user themselves
    sims = sims * (matrix[:, m_idx] > 0)     # only count users who actually rated this movie
    if sims.sum() <= 0:
        return np.nan
    top_k = np.argsort(sims)[-k:]            # pick the k most similar qualifying users
    top_k = top_k[sims[top_k] > 0]
    if len(top_k) == 0:
        return np.nan
    return np.sum(sims[top_k] * matrix[top_k, m_idx]) / np.sum(np.abs(sims[top_k]))


def predict_item_based(u_idx, m_idx, k=20):
    sims = item_sim[m_idx].copy()
    sims[m_idx] = 0                          # exclude the movie itself
    sims = sims * (matrix[u_idx, :] > 0)     # only count movies this user has rated
    if sims.sum() <= 0:
        return np.nan
    top_k = np.argsort(sims)[-k:]
    top_k = top_k[sims[top_k] > 0]
    if len(top_k) == 0:
        return np.nan
    return np.sum(sims[top_k] * matrix[u_idx, top_k]) / np.sum(np.abs(sims[top_k]))

st.subheader("Try a Prediction")
test_user = st.selectbox("Pick a user ID", user_ids)
test_movie = st.selectbox("Pick a movie", movies["title"])

u_idx = uid_to_idx[test_user]
movie_row = movies[movies["title"] == test_movie].iloc[0]
m_idx = mid_to_idx[movie_row["movieId"]]

pred_u = predict_user_based(u_idx, m_idx)
pred_i = predict_item_based(u_idx, m_idx)

st.write(f"User-Based predicted rating: **{pred_u:.2f}**" if not np.isnan(pred_u) else "User-Based: not enough data")
st.write(f"Item-Based predicted rating: **{pred_i:.2f}**" if not np.isnan(pred_i) else "Item-Based: not enough data")

st.subheader("Quick Accuracy Check (RMSE)")

sample = ratings.sample(min(300, len(ratings)), random_state=1)
errs_user, errs_item = [], []

for _, row in sample.iterrows():
    ui, mi = uid_to_idx[row.userId], mid_to_idx[row.movieId]
    true_val = matrix[ui, mi]
    matrix[ui, mi] = 0                      # temporarily hide the real rating
    pu, pi = predict_user_based(ui, mi), predict_item_based(ui, mi)
    matrix[ui, mi] = true_val               # put it back

    if not np.isnan(pu):
        errs_user.append((pu - true_val) ** 2)
    if not np.isnan(pi):
        errs_item.append((pi - true_val) ** 2)

rmse_user_memory = np.sqrt(np.mean(errs_user))
rmse_item_memory = np.sqrt(np.mean(errs_item))

st.write(f"User-Based CF RMSE: **{rmse_user_memory:.4f}**")
st.write(f"Item-Based CF RMSE: **{rmse_item_memory:.4f}**")

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

st.subheader("Train/Test Split")
st.write(f"Training ratings: {len(train_df)}")
st.write(f"Testing ratings: {len(test_df)}")

train_utility = (train_df.pivot_table(index="userId", columns="movieId", values="rating")
                  .reindex(index=user_ids, columns=movie_ids))

user_mean = train_utility.mean(axis=1)
centered = train_utility.sub(user_mean, axis=0).fillna(0).to_numpy()

K = 20  # number of hidden "taste dimensions"

U, S, Vt = np.linalg.svd(centered, full_matrices=False)
P = U[:, :K] * np.sqrt(S[:K])         # user embeddings
Q = Vt[:K, :].T * np.sqrt(S[:K])      # item embeddings

pred_matrix = np.clip(P @ Q.T + user_mean.values.reshape(-1, 1), 0.5, 5.0)

st.subheader("SVD Matrix Factorization")
st.write(f"User embeddings (P): {P.shape}")
st.write(f"Item embeddings (Q): {Q.shape}")
st.write(f"Predicted ratings matrix: {pred_matrix.shape}")

sq_errors_svd = []
for row in test_df.itertuples():
    if row.userId in uid_to_idx and row.movieId in mid_to_idx:
        ui, mi = uid_to_idx[row.userId], mid_to_idx[row.movieId]
        sq_errors_svd.append((pred_matrix[ui, mi] - row.rating) ** 2)

rmse_svd = np.sqrt(np.mean(sq_errors_svd))
st.write(f"SVD RMSE (test set): **{rmse_svd:.4f}**")

st.subheader("What does a hidden dimension capture?")
dim = st.slider("Pick a latent dimension", 0, K-1, 0)

scores = Q[:, dim]
top_idx = np.argsort(scores)[-5:][::-1]
bottom_idx = np.argsort(scores)[:5]

movies_indexed = movies.set_index("movieId")
st.write("**Highest-scoring movies on this dimension:**")
st.write(movies_indexed.loc[movie_ids[top_idx], "title"].tolist())

st.write("**Lowest-scoring movies on this dimension:**")
st.write(movies_indexed.loc[movie_ids[bottom_idx], "title"].tolist())

st.header("Phase 4: Final Evaluation")

comparison = pd.DataFrame({
    "Model": ["User-Based CF", "Item-Based CF", "SVD (Matrix Factorization)"],
    "RMSE": [rmse_user_memory, rmse_item_memory, rmse_svd]
}).sort_values("RMSE")

st.subheader("RMSE Comparison")
st.dataframe(comparison)

fig3, ax3 = plt.subplots()
sns.barplot(data=comparison, x="Model", y="RMSE", hue="Model", palette="viridis", legend=False, ax=ax3)
plt.xticks(rotation=15)
st.pyplot(fig3)

st.subheader("Precision@10")

def precision_at_k(k=10, threshold=4.0, n_sample_users=100):
    hits, total = 0, 0
    for uid in user_ids[:n_sample_users]:
        ui = uid_to_idx[uid]
        relevant = set(test_df[(test_df.userId == uid) & (test_df.rating >= threshold)].movieId)
        if not relevant:
            continue
        scores = pred_matrix[ui].copy()
        scores[matrix[ui] > 0] = -np.inf
        top_idx = np.argsort(scores)[-k:][::-1]
        hits += len(set(movie_ids[top_idx]) & relevant)
        total += k
    return hits / total if total else float("nan")

p_at_10 = precision_at_k(10)
st.write(f"Precision@10: **{p_at_10:.4f}**")

st.subheader("Get Recommendations")

def recommend_movies(user_id, n=5):
    ui = uid_to_idx[user_id]
    scores = pred_matrix[ui].copy()
    scores[matrix[ui] > 0] = -np.inf   # don't recommend movies they've already rated
    top_idx = np.argsort(scores)[-n:][::-1]
    rec_ids = movie_ids[top_idx]
    return movies.set_index("movieId").loc[rec_ids, "title"].tolist()

demo_user = st.selectbox("Pick a user to get recommendations for", user_ids, key="rec_user")
n_recs = st.slider("How many recommendations?", 3, 10, 5)

if st.button("Recommend"):
    recs = recommend_movies(demo_user, n=n_recs)
    st.write(f"**Top {n_recs} recommendations for User #{demo_user}:**")
    for i, title in enumerate(recs, 1):
        st.write(f"{i}. {title}")

st.subheader("Cold Start: New User Recommendations")

def popularity_fallback(n=5):
    stats = ratings.groupby("movieId").agg(avg_rating=("rating", "mean"), n_ratings=("rating", "size"))
    min_votes = stats["n_ratings"].quantile(0.75)   # only consider reasonably well-rated movies
    qualified = stats[stats["n_ratings"] >= min_votes].sort_values("avg_rating", ascending=False)
    return movies.set_index("movieId").loc[qualified.head(n).index, "title"].tolist()

st.write("If a user has no rating history, we can't personalize yet — so we recommend popular, well-reviewed movies instead:")
for i, title in enumerate(popularity_fallback(), 1):
    st.write(f"{i}. {title}")
            