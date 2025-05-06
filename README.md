# Airline_Twitter_Data_sentiment_analysis
I have explained the full process of implementing deep learning models to perform sentiment analysis on airline-related Twitter data. The goal was to classify tweets as positive, neutral, or negative using two main models: LSTM and BERT.

The work started by designing a simple system architecture, which involved the following main stages: data collection, data preprocessing, preparing the data for deep learning, training the models, and finally evaluating the models. Each stage was carefully planned and executed to make sure that the data and models were handled properly.

I collected the dataset from Kaggle, which contained over 14,000 tweets related to major US airlines. Each tweet had a labelled sentiment. In the data preprocessing step, I cleaned the tweets by removing URLs, mentions, hashtags, numbers, and special characters. All text was converted to lowercase, and common stopwords were removed. I also applied stemming to reduce words to their root form. After cleaning, the text was tokenized and padded to prepare it for deep learning models.

In preparing the data, I split it into training and testing sets using an 80/20 split, keeping the balance of sentiments the same in both sets. I applied different tokenization methods for the two models. For LSTM, I used the Keras Tokenizer to convert text into sequences of integers. For BERT, I used Huggingface's special BERT tokenizer, which creates token IDs and attention masks.

The first model I built was the LSTM model. It included an Embedding layer to turn words into vectors, a Bidirectional LSTM layer to capture context from both directions, a Dropout layer to avoid overfitting, and a final Dense layer with a Softmax activation to classify into three categories. I trained the LSTM model over five epochs with a batch size of 32. A small issue was encountered regarding the 'input_length' parameter, which was easily solved by removing it following Keras’ new standards.

The second model was based on BERT. I fine-tuned a pre-trained 'bert-base-uncased' model. This involved special tokenization and a careful setup of the optimizer and learning rate. I used a small learning rate of 2e-5, and compiled the model using Huggingface’s create_optimizer function. The BERT model was trained for three epochs with a batch size of 16. Some technical challenges appeared due to BERT’s large memory usage and compilation requirements, but I managed to solve them by reducing batch size and correctly setting up the optimizer.

Throughout the implementation, I faced and solved several difficulties. Examples include resolving NLTK resource errors when downloading tokenizers, adjusting the model saving format to .keras to match new TensorFlow standards, and correctly handling the optimizer setup for BERT. These challenges improved my technical knowledge and practical understanding of working with deep learning models and libraries.

# ABSTRACT
The rise of social media platforms such as Twitter has revolutionised the way customers express feedback, particularly in the airline industry where real-time opinions can directly influence brand perception. This project explores the application of deep learning methods to automate sentiment classification of airline-related tweets. The core problem addressed was the need to accurately detect customer sentiments negative, neutral, or positive on social media, using natural language processing (NLP) techniques. A key motivation was the challenge of capturing contextual nuances, sarcasm, and informal language patterns in tweets.The evaluation included confusion matrices, classification reports, and training-validation accuracy/loss curves. While BERT outperformed LSTM overall, LSTM was faster and more hardware-efficient, making it suitable for environments with limited resources. The comparison highlighted a key trade-off between performance and efficiency.
This study contributes to sentiment analysis literature by offering a clear, reproducible comparison of LSTM and BERT for tweet classification. It also demonstrates that while BERT excels in contextual understanding, LSTM remains valuable in practical, resource-constrained settings. Future work could include hybrid models, sarcasm detection, and incorporating multimodal data to further improve sentiment classification accuracy and relevance.

#OBJECTIVES
1.	To collect and analyse airline-related tweets for sentiment classification. This involves utilizing the Twitter US Airline Sentiment Dataset, which includes customer feedback directed at major airlines.
2.	To develop a deep learning-based sentiment analysis model. The study will implement models such as LSTM and BERT to classify tweets as positive, negative, or neutral.
3.	To evaluate deep learning models with traditional sentiment analysis techniques. The study will evaluate the accuracy, precision, recall, and F1-score of deep learning models against classical machine learning models such as Support Vector Machines (SVM) and Naïve Bayes.
![image](https://github.com/user-attachments/assets/f764a721-0969-4f35-a449-9bde5b2b39f0)


