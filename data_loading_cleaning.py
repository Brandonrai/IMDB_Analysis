
import pandas as pd

# Load the dataset
data = pd.read_csv('/path/to/IMDB.csv')

# Cleaning the dataset
data['Name'] = data['Name'].apply(lambda x: " ".join(x.split()[1:]))
data['Episodes'] = data['Episodes'].apply(lambda x: int(x.split()[0]))
data['Year'] = data['Year'].apply(lambda x: int(x.split('–')[0]) if '–' in x else int(x))

# Fill missing values in the 'Type' column with the mode
data['Type'].fillna(data['Type'].mode()[0], inplace=True)

# Save cleaned data to a new CSV file
data.to_csv('/path/to/cleaned_IMDB.csv', index=False)
