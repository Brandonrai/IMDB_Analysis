
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import string

# Load the cleaned dataset
data = pd.read_csv('/path/to/cleaned_IMDB.csv')

# Basic list of English stopwords
stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'was', 'he', 'for', 'it', 'with', 'as', 'his', 'on', 'be', 'at', 'by', 'i', 'this', 'had', 'not', 'are', 'but', 'from', 'or', 'have', 'an', 'they', 'which', 'one', 'you', 'were', 'her', 'all', 'she', 'there', 'would', 'their', 'we', 'him', 'been', 'has', 'when', 'who', 'will', 'more', 'if', 'no', 'out', 'so', 'up', 'what', 'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new', 'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first', 'any', 'my', 'now', 'such', 'like', 'other', 'our', 'over', 'more', 'these'])

# Text preprocessing
def preprocess_text(text):
    # Remove punctuation
    text = "".join([word for word in text if word not in string.punctuation])
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text

# Preprocess the description
data['Description'] = data['Description'].apply(lambda x: preprocess_text(x))

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(data['Description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of TV show names and DataFrame indices
indices = pd.Series(data.index, index=data['Name']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the TV show that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all TV shows with that TV show
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the TV shows based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar TV shows
    sim_scores = sim_scores[1:11]

    # Get the TV show indices
    tv_show_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar TV shows
    return data['Name'].iloc[tv_show_indices]

# Test the recommendation system
print(get_recommendations('Breaking Bad'))
