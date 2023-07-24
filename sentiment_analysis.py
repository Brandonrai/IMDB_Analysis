
import pandas as pd
from textblob import TextBlob

# Load the cleaned dataset
data = pd.read_csv('/path/to/cleaned_IMDB.csv')

# Function to get the polarity of a text
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Apply the function to the 'Description' column
data['Sentiment'] = data['Description'].apply(get_polarity)

# Save the data with sentiment to a new CSV file
data.to_csv('/path/to/sentiment_IMDB.csv', index=False)
