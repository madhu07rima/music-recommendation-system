import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('spotify_dataset1.csv')

# Select relevant features for clustering
features = ['popularity', 'energy', 'valence', 'tempo', 'track_genre']
X = df[features]

# Encode the 'track_genre' categorical feature
label_encoder = LabelEncoder()
X['track_genre'] = label_encoder.fit_transform(X['track_genre'])

# Convert features to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scatter plot before clustering
plt.figure(figsize=(10, 6))
plt.scatter(df['popularity'], df['energy'], alpha=0.5, c='blue', label='Data Points')
plt.xlabel('Popularity')
plt.ylabel('Energy')
plt.title('Data Distribution Before Clustering')
plt.legend()
plt.show()

inertias = []
silhouette_scores = []

for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    clusters = kmeans.predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 15), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(2, 15), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different k Values')
plt.grid(True)
plt.show()

optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Get the coordinates of the cluster centers
cluster_centers = kmeans.cluster_centers_

# Scatter plot after clustering
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['popularity'], df['energy'], c=df['cluster'], cmap='tab10', alpha=0.6)
plt.xlabel('Popularity')
plt.ylabel('Energy')
plt.title('Data Distribution After Clustering')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Analyze the clusters
numeric_features = ['popularity', 'energy', 'valence', 'tempo']
cluster_means = df.groupby('cluster')[numeric_features].mean()
print(cluster_means)

def recommend_songs(cluster_centers, target_features):
    distances = [np.linalg.norm(target_features - center) for center in cluster_centers]
    closest_cluster = distances.index(min(distances))
    recommendations = df[df['cluster'] == closest_cluster]['track_name'].tolist()
    return recommendations

import tkinter as tk
from tkinter import messagebox

# Create the main window
root = tk.Tk()
root.title("Spotify Song Recommender")

# Create and place the widgets
tk.Label(root, text="Popularity:").grid(row=0, column=0, padx=10, pady=5)
popularity_entry = tk.Entry(root)
popularity_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Energy:").grid(row=1, column=0, padx=10, pady=5)
energy_entry = tk.Entry(root)
energy_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Valence:").grid(row=2, column=0, padx=10, pady=5)
valence_entry = tk.Entry(root)
valence_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Tempo:").grid(row=3, column=0, padx=10, pady=5)
tempo_entry = tk.Entry(root)
tempo_entry.grid(row=3, column=1, padx=10, pady=5)

genres = ['pop', 'rock', 'hip-hop', 'jazz', 'classical', 'country', 'black-metal', 'bluegrass','blues','sad', 'breakbeat', 'british', 'cantopop','romance','children','chill','sleep', 'club','comedy','reggae','dance','dancehall','death-metal','deep-house','detroit-techno','disco','disney','drum-and-bass','dub','dubstep','edm','electro','electronic','emo','folk']
tk.Label(root, text="Track Genre:").grid(row=4, column=0, padx=10, pady=5)
genre_var = tk.StringVar(root)
genre_var.set(genres[0])  # Set default value
genre_menu = tk.OptionMenu(root, genre_var, *genres)
genre_menu.grid(row=4, column=1, padx=10, pady=5)

def on_recommend_button_click():
    try:
        popularity = float(popularity_entry.get())
        energy = float(energy_entry.get())
        valence = float(valence_entry.get())
        tempo = float(tempo_entry.get())
        track_genre = genre_var.get()
        
        target_features = np.array([popularity, energy, valence, tempo, label_encoder.transform([track_genre])[0]])
        recommended_songs = recommend_songs(cluster_centers, target_features)
        
        messagebox.showinfo("Recommendation", f"Recommended Songs:\n{recommended_songs}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid values for all fields.")

recommend_button = tk.Button(root, text="Recommend Songs", command=on_recommend_button_click)
recommend_button.grid(row=5, column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
