
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset with sentiment
data = pd.read_csv('/path/to/sentiment_IMDB.csv')

# Features
X = data.drop(['Name', 'Image-src', 'Description', 'Name-href', 'Rating'], axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit the model
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the dataframe
data['Cluster'] = labels

# Save the data with clusters to a new CSV file
data.to_csv('/path/to/clusters_IMDB.csv', index=False)

# Set the style of the plots
sns.set_style("whitegrid")

# Plot the distribution of TV shows across the clusters
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Cluster', palette='viridis')
plt.title('Distribution of TV Shows Across Clusters', fontsize=15)
plt.show()
