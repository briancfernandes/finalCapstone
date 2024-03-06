# This python program performs sentiment analysis on a dataset of product 
# reviews from Amazon

# Import packages
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")


# Load a dataset of product reviews from Amazon
df = pd.read_csv("amazon_product_reviews.csv")
df.head()

# Check all the columns in the dataset
df.columns

# Check the length of the dataset
len(df)


# Perform data cleaning on the reviews text by selecting the column 
# containing the reviews and removing any rows with missing data. 
# Save this data in reviews_data.
reviews_data = df['reviews.text'].dropna()
len(reviews_data)


# Create a function to remove stopwords by tokenizing the text and 
# checking if each token is a stopword.
def remove_stopwords(text):
        doc = nlp(text)
        tokens_without_stopwords = [token.text for token in doc 
                                    if not token.is_stop]
        return ' '.join(tokens_without_stopwords)


# Apply the remove_stopwords function to each review in the reviews.txt 
# column and store in a variable reviews_without_stopwords.
reviews_without_stopwords = reviews_data.apply(remove_stopwords)
print(reviews_without_stopwords)


# Create a function that takes a product review as input and predicts 
# its sentiment. A polarity score of 1 indicates a very positive sentiment, 
# while a polarity score of -1 indicates a very negative sentiment. A 
# polarity score of 0 indicates a neutral sentiment.
from textblob import TextBlob

def predict_sentiment(review):

        # Preprocess the review by lowering the text and removing punctuation
        review_text = review.lower().strip().replace(",","")

        # Analyse the sentiment using TextBlob
        blob = TextBlob(review_text)
        polarity = blob.sentiment.polarity

        # Determine the sentiment based on polarity
        sentiment_label = "Neutral"
        if polarity > 0.05:
            sentiment_label = 'Positive'
        elif polarity < -0.05:
            sentiment_label = 'Negative'
  
        return polarity, sentiment_label


# Test the model on 3 random sample reviews. We can use a For loop 
# bounded by the maximum number of rows in the dataset to show the outputs 
# for the reviews.
import random 

random_indices = random.sample(range(len(reviews_data)), 3)

for i in random_indices:
        amazon_reviews_original = reviews_data[i]
        amazon_reviews_clean = reviews_without_stopwords[i]
        polarity, sentiment_label = predict_sentiment(amazon_reviews_clean)

        print(f"Review: {amazon_reviews_original}")
        print(f"Polarity: {polarity:.2f}")
        print(f"Sentiment: {sentiment_label}")
        print("\n")


# We can also run this model on all reviews to get a summary of the
# polarity of the dataset
        
# Initialize counters for sentiment categories
positive_count = 0
negative_count = 0
neutral_count = 0

# Process 5000 reviews
for review in reviews_without_stopwords[:5000]:

        # Predict sentiment
        polarity, sentiment_label = predict_sentiment(review)

        # Update counters
        if sentiment_label == "Positive":
                 positive_count += 1
        elif sentiment_label == "Negative":
                negative_count += 1
        else:
                neutral_count += 1

# Summarize the results
print("Sentiment analysis summary:")
print(f"Positive reviews: {positive_count}")
print(f"Negative reviews: {negative_count}")
print(f"Neutral reviews: {neutral_count}")
print(f"Total reviews analyzed: {len(reviews_without_stopwords[:5000])}")


# Choose two product reviews and compare their similarity. A similarity 
# score of 1 indicates that the two reviews are more similar, while a 
# similarity score of 0 indicates the two reviews are not similar.

# To calculate the similarity we create a function to process the text 
# with nlp while removing punctuation and stopwords.
def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct 
                  and not token.is_stop]
        return tokens


# We can create a function to calculate the similarity between two reviews.
def get_similarity(review1, review2):
    
        # Preprocess the reviews using the first function created above.
        tokens1 = preprocess_text(review1)
        tokens2 = preprocess_text(review2)

        # Create objects from the preprocessed tokens
        doc1 = nlp(' '.join(tokens1))
        doc2 = nlp(' '.join(tokens2))

        # Compute the similarity between the documents
        similarity_score = doc1.similarity(doc2)
        return round(similarity_score, 2)


# For this example we will take the first and second reviews in the dataset.
review1 = df['reviews.text'][0]
review2 = df['reviews.text'][1]
review_similarity = get_similarity(review1, review2)

# Print the sections and the similarity score
print(review1)
print()
print(review2)
print()
print("Similarity between the reviews:", review_similarity)


# A brief report of this analysis is provided in the accompanying PDF file.