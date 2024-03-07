---
title: Capstone Project - NLP Applications readme
---


## Description of the project


* **What is this project?**
The python application in this Capstone Project on NLP Applications performs sentiment analysis on a dataset of product reviews from amazon.

The dataset has been downloaded from Kaggle via the link below: 
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products

It contains the reviews from 5000 users and has 24 columns or fields such as the ID, date added and name of product. The reviews cover 23 unique products largely of either Amazon Fire, Amazon Kindle or Amazon Echo.


* **How does it work?**
The application performs the following steps on the product reviews dataset:
1. The field used for this analysis is "reviews.text". Any rows with missing data are removed and stopwords are removed for the text. Stopwords are words that don't
   add a lot of meaning to the sentence such as 'the' or 'is'.
2. A function is created using TextBlob to take the reviews without stopwords and apply a polarity score to them to give an indication of sentiment.  A polarity 
   score of 1 indicates a very positive sentiment, while a polarity score of -1 indicates a very negative sentiment. A polarity score of 0 indicates a neutral 
   sentiment.
3. This function is run on all 5000 reviews to give a summary of the sentiment for the users of these products.
4. A separate function was created to calculate similarity scores between two reviews. This was done by first creating a function to tokenise the reviews using 
   nlp and then creating a second function to calculate the similarity scores between the reviews using the similarity function in spaCy. A similarity score of 1 indicates that two reviews are similar, while a similarity score of 0 indicates the two reviews are not similar.


* **Who will use this project?**
Readers who would like to understand the sentiment of the amazon technology products dataset or who would like to use their own set of product reviews to measure the
sentiment of that dataset.


* **What is the goal of this project?** 
The goal is to get a general understanding of consumer views of the different products reviewed and how good they are as a product. Any poorly performing products in
particular may need to be withdrawn or improved upon in subsequent versions taking the feedback received into account.



## Instructions for how to develop, use, and test the code.

The "sentiment_analysis.py" python file and the "amazon_product_reviews.csv" file will both need to be saved in the same folder to run. Running the python file
from a platform such as VSCode or Jupyter Notebook will then pick up the csv file and perform the analysis. Any other csv file with "reviews.text" column of product
reviews could you be used instead to perform sentiment analysis on.



### Credits

Credit is given to Kaggle for the use of their amazon datasets for the creation of this application.


