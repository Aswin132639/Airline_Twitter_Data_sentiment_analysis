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

#VISUALIZATION
![image](https://github.com/user-attachments/assets/fbd54aac-2575-4eae-b631-58ad20bc2100)

Rank	Negative Reason	Number of Tweets
1	Customer Service Issue	2,910
2	Late Flight	1,665
3	Can't Tell	1,190
4	Cancelled Flight	847
5	Lost Luggage	724
6	Bad Flight	580
7	Flight Booking Problems	529
8	Flight Attendant Complaints	481
9	Long Lines	178
10	Damaged Luggage	74

![image](https://github.com/user-attachments/assets/2ac86ad7-1219-476b-a047-a23fb23fcc3b)

![image](https://github.com/user-attachments/assets/f0752af3-6cd8-4857-acbb-c525eff86ee6)

![image](https://github.com/user-attachments/assets/ecd13624-fdbd-4d78-8ef2-31a05bd79215)

![image](https://github.com/user-attachments/assets/8d224585-7209-46d9-a832-ff83e7b05412)

![image](https://github.com/user-attachments/assets/45a33056-2d6e-429b-a4c2-5c6d8fbeced8)

![image](https://github.com/user-attachments/assets/3e8ef90a-1e11-47e0-98fa-561b8ec754bf)

![image](https://github.com/user-attachments/assets/488f8b1b-5898-459c-8277-7399944e1b35)

![image](https://github.com/user-attachments/assets/7a16c67e-9d45-4047-9bec-e3bc7b9202f9)

This performance is at a very computationally costly rate, though. Each epoch for training took upwards of 5,500 seconds (or approximately 92 minutes), making BERT less suitable for computing environments with fewer resources. BERT is more memory-intensive, needs more advanced GPUs, and takes longer to train compared to LSTM, so it is less ideal for rapid prototyping or integration into light applications.
Despite this, BERT's generalization capability, contextual richness, and greater overall accuracy position it particularly well for applications requiring a high degree of accuracy in natural language comprehension. Its performance warrants the additional computational cost where erroneous sentiment tagging may have severe reputational or strategic implications, such as in real-time monitoring of customer feedback in the airline sector.
# MODEL COMPARISION
The comparative analysis of the BERT and LSTM models presents insightful observations of their performance patterns. The two models were both trained using the same pre-processed dataset, divided in an 80:20 proportion, to ensure that there was a fair and uniform basis of comparison. As far as overall accuracy is concerned, the BERT model performed slightly better than the LSTM model 80% vs. 78%, respectively.
Where BERT truly has a discernible edge is in classifying neutral and positive sentiments, which are more subtle and context dependent. This is largely because BERT uses a bidirectional transformer model that can see word relationships in both directions, thus being incredibly good at picking up on subtleties of language and shifts in context such as sarcasm, negation, or mixed emotions. In contrast, LSTM models, even with their memory cell mechanism, tend to falter on such complexities unless specifically designed or large.
Yet, LSTM performed unexpectedly well in detecting negative tweets, with a competitive F1-score of 86%. This may be explained by the fact that negative feedback tends to use more explicit and direct language, which can be picked up by LSTM without requiring intricate contextual hints.
There is a substantial trade-off in terms of computational efficiency. BERT requires significantly more training time and hardware resources, with each epoch taking almost 90 minutes to train, whereas LSTM achieves this in a fraction of the time. In real-time or edge deployment scenarios, especially in airline customer service systems where feedback needs to be analyzed in a timely manner, LSTM provides a better cost-performance ratio. However, for high-stakes or finely nuanced applications such as the tracking of public sentiment during service disruptions BERT provides deeper analytical insight.
Though BERT has superior classification accuracy and contextual understanding, LSTM is still relevant in resource-scarce environments. A preference between them is a matter of weighting usage-case factors: performance vs. efficiency.

#CONCLUSION

The purpose of this research was to compare the performance of two deep learning models Long Short-Term Memory (LSTM) and Bidirectional Encoder Representations from Transformers (BERT) in conducting sentiment analysis on the Twitter US Airline Sentiment Dataset. The overall purpose was to ascertain the effectiveness with which the models could classify tweets into three general classes of sentiments: positive, neutral, and negative. In pursuing this endeavour, the project sought to evaluate the feasibility, effectiveness, and reliability of both models within the field of actual airline brand tracking.
The study commenced with a systematic review of current sentiment analysis methods, covering both traditional machine learning and deep learning models. The growing relevance of social media sentiment to companies, especially such customer-oriented industries as aviation, was a compelling motivation for the research. Twitter public opinion presents both challenges and opportunities to airlines, and automatic detection and classification of such opinion can facilitate rapid customer response and effective reputation management.
Through the development and implementation of LSTM and BERT models, the project managed to attain its set objectives. The models were trained on the cleaned data after rigorous cleaning and tokenisation procedures. Both models were tested with standard classification measures such as accuracy, precision, recall, and F1-score. The LSTM model had an accuracy of 78%, while that of the BERT model was 80% accuracy, which was a bit higher. Specifically, BERT performed more accurately in differentiating between positive and neutral sentiments, which tends to require a richer contextual understanding.
This chapter summarizes the principal findings, discusses future research opportunities, and gives a reflective evaluation of the project process. The report is concluded by giving an overview of the work that was done and its broader implications.

#FUTURE WORK
Although this study yielded significant results and fulfilled its primary objectives, there exist various avenues through which the project may be expanded to develop more robust and scalable sentiment analysis models. The present implementation was confined to textual data from a single social media platform Twitter and pertained specifically to one sector, namely the airline industry. Future research can involve cross-platform sentiment analysis that integrates tweets with reviews from other websites such as Facebook, Instagram, or TripAdvisor to derive a more comprehensive picture of the public opinion.
One aspect that is ready to be improved upon is the addition of sarcasm and emotion detection modules. In BERT and LSTM testing, a lot of the neutral or confusing tweets are mislabelled due to sarcastic tone or contradictory emotions within a single post. Employing multi-task learning methods that would simultaneously train a model to detect both sentiment and sarcasm can significantly improve classification accuracy, especially in the neutral category. One more beneficial aspect of future research is hyperparameter tuning. For the present project, the parameters were fixed, i.e., batch size 32 and 5 epochs for LSTM and batch size 16 and 3 epochs for BERT. Though these parameters performed well, it might be the case that adjusting the learning rate, dropout rate, and units could further improve the model performance.
Automated techniques like Grid Search or Bayesian Optimisation may be utilized to simplify this procedure. Furthermore, the idea of building hybrid models can be investigated in future work. For instance, researchers can try to combine the contextual power of BERT and the sequential learning capacity of BiLSTM. A model that uses BERT embeddings as input to a BiLSTM layer can potentially benefit from both global context and temporal relationships. There is some early work that has suggested that this hybrid model improves classification accuracy on certain NLP tasks, e.g., sentiment analysis.
To enhance scalability without incurring computational overhead, you can use model compression techniques, such as knowledge distillation. Compressed transformer model variants, such as DistilBERT or TinyBERT, offer a tuned trade-off between accuracy and performance and thus are a good match for real-time applications where latency is of paramount importance. Lastly, one other direction of interest is multimodal sentiment analysis. Most Twitter posts comprise images, emojis, and hashtags in addition to text. The inclusion of visual information or metadata (for instance, likes and retweets) could introduce an additional layer of information, which may augment sentiment classification. This would involve the use of computer vision models and multi-input models but would bring the system nearer to how humans receive sentiment through more than a single cue aside from text.











