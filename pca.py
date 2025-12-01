import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from utils import load_file, load_vocab, tweets_to_features

# --- Chargement des données ---
train_pos = "twitter-datasets/train_pos.txt"
train_neg = "twitter-datasets/train_neg.txt"
final_test = "twitter-datasets/test_data.txt"
embedding_vocab_file = "vocab_cut.txt"

tweets_train_pos, tweets_train_neg, tweets_test = load_file(train_pos, train_neg, final_test)
embedding = np.load("embeddings.npy")
embedding_vocab = load_vocab(embedding_vocab_file)

# --- Construction des vecteurs ---
X_pos = tweets_to_features(tweets_train_pos, embedding_vocab, embedding)
X_neg = tweets_to_features(tweets_train_neg, embedding_vocab, embedding)

# --- Normalisation L2 ---
X_pos = normalize(X_pos)
X_neg = normalize(X_neg)

# --- Labels 1=positif, 0=négatif ---
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

X = np.vstack([X_pos, X_neg])
y = np.hstack([y_pos, y_neg])

# --- PCA pour réduire à 2D ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# --- Visualisation ---
plt.figure(figsize=(8,6))
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='green', alpha=0.5, label='Positif')
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', alpha=0.5, label='Négatif')
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection PCA des tweets")
plt.legend()
plt.show()
