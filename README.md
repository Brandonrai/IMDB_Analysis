# IMDb Top 250 TV Shows Analysis and Prediction

This project involves the analysis and prediction of the top 250 TV shows according to IMDb ratings. The tasks performed include exploratory data analysis (EDA), text preprocessing, building a recommendation system, sentiment analysis, regression modeling, and clustering. The dataset used for this project contains 250 unique TV shows with details such as name, release year, number of episodes, show type, IMDb rating, image source link, and a brief description.

## Python Scripts

1. `data_loading_cleaning.py`: This script loads the dataset, cleans the data, and saves the cleaned data to a new CSV file.
2. `eda.py`: This script generates distribution plots for release years and IMDb ratings, and a scatter plot showing the relationship between release year and IMDb ratings.
3. `recommendation_system.py`: This script preprocesses the TV show descriptions, creates a TF-IDF matrix, computes a cosine similarity matrix, and defines a function to get TV show recommendations based on their descriptions. It tests the recommendation system with the TV show "Breaking Bad".
4. `sentiment_analysis.py`: This script performs sentiment analysis on the TV show descriptions and saves the data with the sentiment scores to a new CSV file.
5. `regression_model.py`: This script splits the data into training and testing sets, trains a linear regression model, makes predictions on the testing set, and computes the Mean Squared Error (MSE).
6. `clustering.py`: This script scales the features, applies the K-means clustering algorithm, adds the cluster labels to the DataFrame, saves the data with the clusters to a new CSV file, and generates a bar plot showing the distribution of TV shows across the clusters.

## How to Run the Scripts

1. Clone the GitHub repository.
2. Install the necessary libraries. This can be done by running `pip install -r requirements.txt` in the command line. The `requirements.txt` file should list the required libraries.
3. Run each Python script in the command line using the command `python script_name.py`, replacing `script_name.py` with the name of the script. Make sure to run the scripts in the order they are listed above.
4. View the output CSV files and plots.

## Contributing

Contributions are welcome. Please open an issue to discuss your ideas or open a pull request with your changes.

## License

This project is licensed under the terms of the MIT license.
