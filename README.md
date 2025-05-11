# Dream Analysis Project
**Dreams Analysis Through Classification and Clustering**

**Dual Bachelor in Data Science and Engineering & Telecommunication Technologies Engineering**

**Course & Group:** Machine Learning Applications G. 196  

<p align="center">
  <img src="https://github.com/user-attachments/assets/787e8b0d-438b-49c3-bae3-1f9ba6f97d23" alt="Captura de pantalla 2025-05-11 165252">
</p>


## Team Members

- **Irina Vela Gómez** – 100454302  
- **Claudia Sanchez Merino** – 100475131  
- **Lucía Fernández Alba** – 100475223  
- **Mónica Martín Herguedas** – 100474845


## Report Structure

We have divided the report into the following parts:

1. **Introduction**  
2. **Dataset Preparation**  
3. **Task 1. Natural Language Processing and Text Vectorization**  
    - 3.1 **Text Preprocessing Pipeline**  
    - 3.2 **Text Vectorization**  
4. **Task 2. Machine Learning**  
    - 4.1 **Classification**
    - 4.1 **Clustering**  
5. **Task 3. Implementation of a Dashboard**  
6. **Acknowledgement of Authorship**  
7. **Conclusion**  


## 1. Introduction

In our final project, we use Natural Language Processing (NLP) techniques to study personal dream reports from a dataset of user-submitted information. The techniques that were used in this project are classification and clustering. 

For classification, we wanted to train a model capable of predicting the emotion associated with the dream. In clustering, we aimed to project the dreams and group them into clusters to try and find relations and patterns within the groups.

We also wanted to recommend similar dreams to users, as well as users who dream similar things. However, since dreams are a personal experience and it is impossible for two people to dream exactly the same thing, we were unable to recommend specific dreams.

## 2. Dataset preparation 


At the beginning we started working with DreamBank dataset, which is publicly available in Hugging Face and had 22k entries. It lacked complementary variables like age, emotions… After performing k means clustering with k = 4000 to have a varied dataset, and start working with it we realised we could obtain a poor work, so we decided to change the dataset, but not the scope

For the preparation of the dataset, we selected the DreamBank-annotated dataset, which is also publicly available through Hugging Face, which contains 28k entries. The first stage was to use an initial filter, discarding entries that had missing values and with length of less than 50 words, assuming that shorter dreams are unlikely to contain meaningful content. After this, we used a KMeans clustering algorithm with 4000 clusters, since this is the desired size for our dataset. By choosing the closest dreams to the cluster centroids, we ensured that the set was sufficiently diverse, as each selected dream belongs to a different cluster, guaranteeing a varied representation of the content. This was a strategy that enabled us to minimize the dataset, yet preserve its diversity and relevance which is important for further analysis.

The main advantage of this dataset is that it provided some extra information about the dreams that we could use for the work like: age group, characters in the dream, emotions and who felt them. Our target variable is going to be emotions, but we had to clean it since it contained several labels among AN anger, AP anxiety, SD sadness, CO confusion and HA positive emotions and the person that felt the emotion. We established that D stands for Dreamer and we are going to consider the emotion associated with the dreamer as the main emotion of the dream. We cleaned the year data as well, as it gave a range we decided to keep the oldest year. 


## 3. Task 1. Natural Language Processing and Text Vectorization

### 3.1 Text Preprocessing Pipeline

Text pre-processing of input data is the initial step in our task, and the results we obtain with the project depend directly on this crucial step. Here, we have implemented a complete pipeline for the preprocessing of the texts, which consists of four stages: text wrangling, tokenization, homogenization, and cleaning. The processing uses built-in functions combined with proven NLP tools, including SpaCy, NLTK, and the contractions package to execute each transformation step. We made use of the library SpaCy because we could benefit from its resources for tokenization and part-of-speech (POS) tagging.

#### 1. Text Wrangling
The first step in the preprocessing phase is text wrangling, where we expand contracted negations to their full forms (e.g., don't → do not), using the contractions library. We also remove unwanted symbols such as brackets, quotation marks, and parentheses with regular expressions. Dates that appeared in parentheses (e.g., (12/03/2022)) were also removed using a custom regex-based function, as they were not considered part of the dream content. Additionally, we have a data variable called "year," so having this kind of information in the dream description gives us no additional information. Since our main goal is to cluster dreams based on their thematic and emotional content, we excluded this temporal information to avoid noise.

#### 2. Tokenization
The second step is tokenization, where each token represents a single word. We considered that word-level tokenization is more appropriate for the thematic analysis we are focusing on, rather than sentence-level tokenization. This was done by applying SpaCy’s `en_core_web_md` model to the cleaned text.

#### 3. Homogenization
After that, in the homogenization step, we convert all tokens to lowercase to achieve consistency and avoid treating words like “Dream” and “dream” as different tokens. We also apply lemmatization using SpaCy’s `lemma_` attribute to reduce words to their base forms, allowing us to group together variants like "running," "ran," and "runs" under the lemma "run." We also tried stemming, but as lemmatization is generally more accurate and produces more meaningful outcomes, we decided to use lemmatization.

#### 4. Cleaning
In the cleaning phase, we employ token filtering protocols. We remove common stopwords using the SpaCy attribute `is_stop` and keep only alphabetic tokens that belong to specific parts of speech relevant to our analysis. We decided to keep nouns, verbs, adjectives, and adverbs. With this step, we can focus on semantically rich words that will capture better dream themes. This technique is very important for improving efficiency since we are only keeping meaningful information, and it also reduces noise, making this step crucial for text preprocessing.

It is crucial that the steps are followed in the order we explained them. All of the steps are included in a pipeline, so we only need to execute that function to clean all our data entries.

### 3.2 Text Vectorization

We have applied various text vectorization techniques to our dataset. We already know that text vectorization is the process of converting textual data into numerical vectors that machine learning algorithms can interpret. We investigated different vectorization schemes, from classical ones to more sophisticated semantic embeddings.

#### 1. Classical Vectorization: BoW and TF-IDF

**Bag of Words (BoW):** The Bag of Words vectorization represents documents as sparse vectors based on word frequencies. In our implementation, we restricted the representation to the 5000 most frequent words using scikit-learn’s `CountVectorizer`. The resulting matrix had dimensions corresponding to the number of documents and the vocabulary size. This representation is one of the simplest, as it only counts how many times each word appears in each document, disregarding word order and semantic relationships.

**Term Frequency-Inverse Document Frequency (TF-IDF):** With this vectorization technique, we preserve the idea of the Bag-of-Words model, since it still relies on word frequencies. However, TF-IDF introduces a significant modification by weighting term frequencies by their inverse document frequency, giving more importance to words that are frequent in individual documents but rare in the entire corpus. To implement it, we used scikit-learn’s `TfidfVectorizer`, limiting the model to the top 5,000 features.

We made a visualization of both the BoW and TF-IDF results, using bar charts to observe the rankings. From these graphs, we observe that common words such as ‘go’ and ‘say’ still appear as the top 2 most frequent words in both graphs, although TF-IDF gives them less importance due to their repeated occurrence in the corpus. The overall rankings of these two vectorizations are quite similar, suggesting that the difference between the two methods was not as pronounced as it might be in other contexts.

![4](https://github.com/user-attachments/assets/545a97cf-6764-4db4-af42-52147b2b4c79)

These two  methods have a common limitation, they treat words as isolated units, not taking into account the context or the semantic relationships.

#### 2. Word Embeddings: Word2Vec and Doc2Vec

**Word2Vec:** Word2Vec generates dense, continuous vector representations of words based on their contextual relationships in the text. We implemented Word2Vec with a vector size of 100 dimensions, a context window of 5 words, and ignored words that appeared fewer than twice in the corpus.

To convert these word-level vectors into document vectors, we implemented a document vector function (`document_vector(word_list, model)`) that computes the average of all word vectors in a document. More specifically, for each document, we obtained the subset of words in that document that exists in the trained Word2Vec vocabulary, looked up their embeddings, and averaged over all these vectors. When a document included no contextually recognizable tokens, we set a zero vector, maintaining dimensionality.

This averaging approach is a simple but effective way to create document-level representations. It does not capture word order or complex syntactic structures, but it preserves some of the semantic relationships captured by Word2Vec.

**Doc2Vec:** Unlike Word2Vec, which generates embeddings at the word level, Doc2Vec generalizes this idea and constructs distributed representations of the whole document, i.e., document-level embedding. We implemented Doc2Vec by first tagging each document with a unique identifier and then training the model with the same vector size of 100 dimensions, a minimum word count of 2, and 30 training epochs.

Document vectors were then inferred directly from the model, bypassing the need for the averaging step used with Word2Vec. This approach enables the model to understand document-level semantics more explicitly, as it uses both the context of words and the identification of the documents during training.

Both Word2Vec and Doc2Vec generate dense 100-dimensional representations for documents. They both capture semantic relationships that the classical BoW and TF-IDF methods don't capture. These representations are particularly better for tasks that require understanding word or document similarities beyond simple term frequency, which is exactly what we value in our dream analysis.

#### 3. Topic Modeling with Latent Dirichlet Allocation (LDA)

LDA is a probabilistic model that discovers "topics" in a collection of documents. It provides interpretable topics consisting of words with associated probabilities.

To implement LDA, we first created a dictionary and corpus using Gensim. To determine the optimal number of topics, we implemented a function to compute coherence scores for different numbers of topics, testing the range from 2 to 40 topics (steps of 6) and selecting the number with the highest coherence score.

The coherence scores were plotted against the number of topics. As shown in the following image, the maximum coherence score was achieved at around 20 topics, which was determined to be the optimal number for our dataset. This suggests that 20 topics provide the best balance between model complexity and topic interpretability. The final LDA model was trained with this optimal number of topics.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5dd9918b-6cdb-49d4-8b73-2f3e6e374cb7">
</p>

Each document was then represented as a vector of topic probabilities. This involved obtaining the probability distribution over all topics for each document, with the resulting matrix having dimensions corresponding to the number of documents and the number of topics.

The LDA visualization and topic bar plots were generated to provide a clear interpretation of the discovered topics. Each topic consisted of specific words with associated weights, from which we can obtain some conclusions.

![image](https://github.com/user-attachments/assets/189341c0-637a-4394-bc5d-0c421add1b23)

As mentioned before, the LDA model identified distinct topics in our corpus that seem to correspond with meaningful categories. For example, based on the graph of above:
Topic 0: Words like "game", "team", "play", and "ball" suggest a topic related to sports and team activities.
Topic 3: Words like "friend", "feeling", "life", and "dream" suggest emotional and interpersonal relationships.
Topic 4: Words like "test", "exam", "football", “math”,  and "school" point to a topic including education.
Topic 7: Words like "car", "water", "drive", and "plane" clearly relate to transportation and travel.
Topic 8: Words like "car", "road", "stop", and "drive" focus specifically on road transportation.
These discovered topics provide valuable insights into the main themes present in our corpus and can be used for document classification or recommendation systems.

## 4. Task 2 Machine Learning
### 4.1 Task 2.1. Classification

For classification, our objective was to classify the emotion of a dream. We used the ‘emotion_feature’, which contains the following labels:
- **HA**: positive emotions 
- **AP**: anxiety 
- **SD**: sadness
- **AN**: anger
- **CO**: confusion

We explored various document vectorization techniques such as Bag of Words (BoW), TF-IDF, Word2Vec, Doc2Vec, and Latent Dirichlet Allocation (LDA). The procedure was integrated using pipelines and classifiers such as logistic regression, random forests, and SVM, with hyperparameter tuning (using cross-validation).

Due to the complex, subjective, and often ambiguous nature of dream narratives, as well as the fact that dreams can express multiple emotions simultaneously (even though a single label was forced for each entry), it was expected that the models would not achieve high accuracy scores. Performance evaluation metrics included accuracy, weighted F1-score, ROC-AUC, and confusion matrices. We centered on analyzing the ROC, as it is a more robust and reliable metric than accuracy.

The following table shows the ROC scores of the models:

| Model             | Logistic Regression | Random Forest | SVM   |
|-------------------|---------------------|---------------|-------|
| **BoW**           | 0.628563            | 0.621325      | 0.609014 |
| **TF-IDF**        | 0.638461            | 0.609786      | 0.621993 |
| **Word2Vec**      | 0.626912            | 0.570033      | 0.628758 |
| **Doc2Vec**       | 0.696915            | 0.652015      | 0.711998 |
| **LDA**           | 0.612340            | 0.589181      | 0.552758 |

We can see that the best results were obtained with **Doc2Vec**, which makes sense. Doc2Vec generates vector representations that capture the overall meaning of an entire text, rather than focusing solely on word frequency and co-occurrence (BoW, TFIDF) or individual words (Word2Vec). This is particularly relevant in texts such as dream descriptions, where emotional content often depends on a sequence of events and symbolic language.

LDA, which is also a very powerful method, did not perform well here. This could be because LDA is better suited for unsupervised tasks and is a topic modeling approach that does not effectively capture emotional connotations. As a result, it produces representations that are not informative enough for classifiers in this type of task.

To assess whether the models were capable of learning meaningful patterns from the text in a more controlled setting (not focused on emotions), a second, simpler experiment was designed: binary gender classification based on the dream content (male vs. female author). The results were notably better:

| Model             | Logistic Regression | Random Forest | SVM   |
|-------------------|---------------------|---------------|-------|
| **BoW**           | 0.932132            | 0.986506      | 0.888932 |
| **TF-IDF**        | 0.936908            | 0.923538      | 0.950534 |

The results were notably stronger, even with just BoW and TF-IDF. These results suggest that dreams are more directly related to gender based on the language used within dream reports, making the classification task more straightforward for machine learning models.

The performance in this binary classification task confirms that the models are capable of extracting discriminative features from the dream texts and reinforces the idea that the lower performance in emotion classification is not due to model incapacity, but rather due to the complexity, ambiguity, and subjectivity of emotional expressions.

### 4.2 Task 2.2. Clustering

We thought it would be interesting to analyze the themes of dreams and see if they were related or if we could identify predominant themes.

To achieve this, the **K-Means** algorithm was applied to the **LDA matrix**, which contains the topic distributions per document (probabilities indicating how strongly each dream is associated with each topic). This approach was ideal because the LDA representation is normalized and continuous, and all observations are on a comparable scale (useful since K-Means works with Euclidean distance).

To obtain the optimal number of clusters, we used a combination of **Silhouette Score** and **Elbow Method**. In the graphs below, we cannot see any clear elbow, and the silhouette values are quite low, indicating that the clusters have a moderate quality. By combining both graphs, we can clearly set the optimal number of clusters at **7**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/25c7b128-4793-4dbe-85e4-12c387161ca0">
</p>

To visualize the clusters, we apply PCA, t-SNE, and UMAP. These dimensionality reduction techniques reduce the data to 2D or 3D space, helping us visualize the distribution and separation of the clusters, making it easier to interpret the results of K-Means clustering.

These following plots show the relationship between the LDA topics and the K-Means clusters. In the left-hand visualization, each dream is colored according to its predominant topic, meaning the topic with the highest probability in its LDA representation. In the right-hand visualization, the colors indicate the clusters assigned by K-Means, which were learned based on the topic distributions produced by the LDA model (7 clusters have been used in both plots).

<p align="center">
  <img src="https://github.com/user-attachments/assets/b3c8854e-da2c-450d-93fc-aa4c214e6b3e">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5679b5d-ad8d-4611-a9cb-5066629a3906">

In this last plot, using K-Means with 20 clusters, the Davies-Bouldin index ranged from 1.679 (which is the index obtained with 7 clusters) to 1.638, which does not indicate a significant improvement. This range suggests that the clusters are not perfectly separated and compact, but they are also not completely dispersed or poorly defined. Quite reasonable results given a real dataset. 

By comparing the relation of topics and the two cluster representations, we can see that with 7 clusters we represent the bigger topics, meaning the topics that have more entries, but if we check the 20 clusters graph we can appreciate that it is capturing the smaller topics, the topics that have less number of dreams associated.







While no extremely clear divisions are observed among the clusters, there are several notable differences:

- **Based on numerical statistics by cluster, some groups are gender-skewed:**
  - Clusters 0, 1, and 2 have a majority of female participants.
  - Clusters 3 and 4 are more gender-balanced, particularly cluster 4.
  - Clusters 5 and 6 are smaller in size, but also tend to have more female participants.

- **Regarding the distribution of emotions per cluster:**
  - The **AP** emotion (fear, anxiety) is the most common across all clusters.
  - Cluster 1 has the highest absolute number of emotions overall.
  - Cluster 5, despite being small, contains a relatively high number of **happy dreams (HA)**.
  - Cluster 6 is the smallest in terms of size and also shows the least emotional diversity.

![image](https://github.com/user-attachments/assets/6af488e4-7a94-422a-8c1d-b4041179a45c)






## 5. Task 3. Implementation of a dashboard




## 6. Acknowledgement of authorship
This project has been developed by the four of us. We have been working collaboratively in the structure, development, and analysis of the different tasks and results. We have discussed which tools and methods to implement are the best options to address the challenges faced throughout the process.
During the implementation, we found technical and conceptual obstacles and issues that led us to explore and experiment. When we got stuck or when the results were unclear, we used artificial intelligence, such as ChatGPT, Gemini, and Claude. These tools facilitated our analysis, and gave us some ideas on how to improve our approach. Nevertheless, it did not solve all of our questions and sometimes it gave wrong answers, so we had to be careful and verify things twice in order to not make mistakes.

## 7. Conclusion
With this project, we conclude the course of Machine Learning Applications. Throughout the development, we consider that we have applied most of the knowledge gained on it, such as feature engineering or Natural Language Processing techniques.

It was an interesting project since we have tried by ourselves in an independent way to achieve results and how to get them. It  also has to be said that it has been a hard challenge. We encountered a variety of issues and problems, such as compatibility errors between libraries and packages, difficulties interpreting results, or unexpected insights that made us change how to focus the tasks.

We have discovered that not everything is as easy and perfect as it might initially seem to be.  In many academic projects, we worked with prepared datasets that, of course, are going to fit perfectly with the tasks proposed. Nevertheless, with the dataset selected about dreams, we realized that it may be difficult to achieve high accuracy,  clear distinctions in classification, or meaningful clustering, and that the analysis can become more complex when dealing with real, messy, and subjective data. This experience taught us the importance of adaptability, critical thinking, and the value of working through uncertainty in the real world data analysis.

To put it simply and succinctly , while the development of it was full of unexpected turns and demanding moments, it has been an enriching experience from which we have learned a lot. We would have loved to achieve perfect results and ideal and clear conclusions. However, in real life this is not so straightforward and we are proud of what we’ve achieved and of the knowledge acquired. 


