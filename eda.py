
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv('/path/to/cleaned_IMDB.csv')

# Set the style of the plots
sns.set_style("whitegrid")

# Create a figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(14, 10))

# Plot the distribution of release years
sns.histplot(data=data, x='Year', bins=30, ax=ax[0], color='skyblue', kde=True)
ax[0].set_title('Distribution of TV Shows by Release Year', fontsize=15)

# Plot the distribution of IMDb ratings
sns.histplot(data=data, x='Rating', bins=30, ax=ax[1], color='skyblue', kde=True)
ax[1].set_title('Distribution of TV Shows by IMDb Ratings', fontsize=15)

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()

# Plot the relationship between Year and Rating
plt.figure(figsize=(14, 6))
sns.scatterplot(data=data, x='Year', y='Rating', alpha=0.7)
plt.title('IMDb Ratings vs. Release Year', fontsize=15)
plt.show()
