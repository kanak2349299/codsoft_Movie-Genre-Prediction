**🎬 Movie Genre Prediction using Machine Learning**

# **📌 Overview**
Movie Genre Prediction is an NLP and Machine Learning project that predicts the genre of a movie from its textual information.
The project uses the Genre Classification Dataset IMDb from Kaggle. Year and Description are used as input features, while Genre is the target variable.
Three classification algorithms were trained and compared:
- *🧠 Logistic Regression*
- *⚡ Linear SVM*
- *📊 Multinomial Naive Bayes*
Logistic Regression achieved the best performance on the test data and was selected as the final model.


# **🎯 Problem Statement**
Develop a Machine Learning model that can automatically predict a movie's genre based on its year and description.
Manual genre classification can be time-consuming for large movie databases. This project uses NLP and Machine Learning to automate the classification process.


# **💡 Objectives**

- 🎥 Analyze movie genre data
- 🧹 Clean and preprocess textual information
- 📊 Perform exploratory data analysis
- ☁️ Generate genre-specific Word Clouds
- 🔀 Perform train-test splitting
- 📝 Convert text into numerical features using TF-IDF
- 🤖 Train multiple classification models
- 📈 Compare model performance
- 🏆 Select the best model
- 🎬 Predict the genre of a new movie description
- 📂 Dataset
- Genre Classification Dataset IMDb
  
Source: Kaggle
Domain: Movies and TV Shows 🎥
The original dataset contains:
description.txt
train_data.txt
test_data.txt
test_data_solution.txt
The prepared data used in this project contains:

Column           Role     Description
📅 Year          Input    Movie year / year information 
📝 Description   Input    Movie plot or description
🎭 Genre         Target   Genre to be predicted

*📌 The dataset is not included in this repository because of its size. Download it from Kaggle before running the notebook.*


# **🧠 Project Workflow**
📂 Dataset
   ↓
🧹 Data Cleaning
   ↓
🔤 Text Preprocessing
   ↓
📊 EDA
   ↓
🔀 Train-Test Split
   ↓
📝 TF-IDF
   ↓
🤖 Model Training
   ├── Logistic Regression
   ├── Linear SVM
   └── Multinomial Naive Bayes
   ↓
📈 Model Evaluation
   ↓
🏆 Best Model Selection
   ↓
🎬 New Movie Prediction


# **🧹 Data Preprocessing**
The text preprocessing pipeline includes:
🔡 Convert text to lowercase
✂️ Remove punctuation
🚫 Remove English stop words
🔤 Tokenize text
🌱 Apply Porter Stemmer
🔗 Join the processed tokens

*The same preprocess() function is used for training data and new movie descriptions.*
Example
Playing → play
Played  → play
Plays   → play
**ℹ️ Punctuation removal does not mean that numeric information is intentionally removed.**


# **📊 Exploratory Data Analysis**
EDA was performed to understand the dataset and genre distribution.
Visualizations

- 📊 Genre distribution bar plot
- ☁️ Action Word Cloud
- ☁️ Comedy Word Cloud
- ☁️ Drama Word Cloud
- ☁️ Horror Word Cloud
- 🔥 Confusion Matrix
*Word Clouds help visualize frequently occurring words within movie descriptions for selected genres.*


# **🔀 Train-Test Split**
The cleaned input and target variables are divided into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
🏋️ Training data → used to train the models
🧪 Testing data → used to evaluate unseen examples

The same split is used for all three models for a fair comparison.


# **📝 TF-IDF Feature Extraction**
Raw text cannot be directly supplied to traditional Machine Learning models. Therefore, TF-IDF is used to convert text into numerical features.
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 1)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# **⚠️ Data Leakage Prevention**
The vectorizer is fitted only on training data:
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
It should not be fitted on the complete dataset before splitting.

# **📌 Scaling**
*Additional scaling is not required after TF-IDF for the text classification models used in this project.*


# **🤖 Machine Learning Models**
**🥇 Logistic Regression**
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

Logistic Regression achieved the best overall performance in the performed comparison.

**⚡ Linear SVM**
from sklearn.svm import LinearSVC
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
Linear SVM is well suited to high-dimensional sparse TF-IDF features.

**📊 Multinomial Naive Bayes**
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
Multinomial Naive Bayes provides a strong and efficient baseline for text classification.


# **🏆 Model Comparison**
All models use the same:
🔀 Train-test split
📝 TF-IDF representation
🎯 Target labels
🧪 Test set

**Ranking*
Rank   Model                         Result
🥇 1   Logistic Regression       ⭐ Best 
🥈 2   Linear SVM                    Very Good 
🥉 3   Multinomial Naive Bayes       Lower than the other two

**Final Model*
🏆 Logistic Regression was selected because it achieved the best overall performance on the dataset.

The exact scores depend on the dataset version, preprocessing, TF-IDF parameters, and train-test split.


# **📏 Evaluation Metrics**
The models are evaluated using:
🎯 Accuracy
🔍 Precision
🔄 Recall
⭐ F1-score
📊 Macro F1-score
🔥 Confusion Matrix
For multi-class classification, macro F1-score is especially useful because it gives equal importance to individual classes.

**🔥 Confusion Matrix**
The confusion matrix shows actual genres against predicted genres and helps identify:
✅ Correct classifications
❌ Incorrect classifications
🔄 Genres frequently confused with each other


# **🎬 Final Prediction**
After selecting Logistic Regression, a new movie description can be classified.
movie_description = "A detective investigates a mysterious murder and discovers a dangerous criminal gang behind it."
cleaned_movie = preprocess(movie_description)
movie_tfidf = tfidf.transform([cleaned_movie])
prediction = lr_model.predict(movie_tfidf)
predicted_genre = le.inverse_transform(prediction)
print("Predicted Genre:", predicted_genre[0])


# **🔄 Prediction Pipeline**

📝 New Movie Description
        ↓
🧹 preprocess()
        ↓
📝 Fitted TF-IDF Vectorizer
        ↓
🧠 Logistic Regression
        ↓
🔢 Encoded Genre
        ↓
🏷️ Label Encoder
        ↓
🎭 Original Genre
        ↓
💾 Saved Model Files

The repository contains the trained components:
File                     Purpose
🤖 model.pkl           Trained Logistic Regression model 
🔤 tfidf.pkl           Fitted TF-IDF vectorizer 
🏷️ label_encoder.pkl   Genre Label Encoder

They can be loaded later without retraining.
import pickle
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
with open("tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)
with open("label_encoder.pkl", "rb") as file:
    le = pickle.load(file)

# 📁 Repository Structure
Movie-Genre-Prediction/
│
├── 📓 Genre_classify.ipynb
├── 🤖 model.pkl
├── 🔤 tfidf.pkl
├── 🏷️ label_encoder.pkl
│
├── 📊 Screenshots/
│   ├── bar_plot.png
│   ├── confusion_matrix.png
│   ├── action_wordcloud.png
│   ├── comedy_wordcloud.png
│   ├── drama_wordcloud.png
│   └── horror_wordcloud.png
│
├── 📄 README.md
└── 📜 .gitignore


# **🛠️ Tech Stack**
- 💻 Programming
- 🐍 Python
- ☁️ Google Colab / Jupyter Notebook
- 📚 Libraries
- 🐼 Pandas
- 🔢 NumPy
- 📊 Matplotlib
- 📈 Seaborn
- 📝 NLTK
- 🤖 Scikit-learn
- ☁️ WordCloud
- 🧠 NLP & ML

# Text preprocessing
- Tokenization
- Stop-word removal
- Porter Stemming
- TF-IDF Vectorization
- Label Encoding
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

# 🚀 How to Run

1️⃣ Clone the repository
git clone <your-github-repository-link>
2️⃣ Install dependencies
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
3️⃣ Download the Dataset
Download the Genre Classification Dataset IMDb from Kaggle and place the required files in the appropriate location.
4️⃣ Open the Notebook
Open:
Genre_classify.ipynb
using Google Colab or Jupyter Notebook.
5️⃣ Run the Notebook
Run the cells in order:
Data Loading
↓
Cleaning
↓
Preprocessing
↓
EDA
↓
Train-Test Split
↓
TF-IDF
↓
Model Training
↓
Model Comparison
↓
Evaluation
↓
Final Prediction

# ⚡ Performance Considerations
TF-IDF can generate a large sparse matrix. To reduce memory usage and training time, the number of features can be limited:
TfidfVectorizer(max_features=20000)
This provides a practical balance between computational efficiency and useful text features.

# 📚 Learning Outcomes
This project demonstrates:
✅ Data cleaning
✅ Exploratory Data Analysis
✅ Natural Language Processing
✅ Text preprocessing
✅ Stop-word removal
✅ Porter stemming
✅ TF-IDF feature extraction
✅ Train-test splitting
✅ Multi-class classification
✅ Model comparison
✅ Accuracy and F1 evaluation
✅ Confusion matrix analysis
✅ Model serialization
✅ New-text prediction

# **⚠️ Limitations*

🎭 Some genres may contain fewer examples than others.
📝 Prediction depends on the quality of the movie description.
🌱 Stemming can sometimes produce less readable word forms.
🎬 A movie may have multiple genres, while the dataset's target is treated as a classification label.
🤖 Traditional ML models have less contextual understanding than modern Transformer models.
🔮 Future Improvements
🧠 Word2Vec or GloVe embeddings
🤖 BERT / Transformer-based classification
🎯 Hyperparameter tuning
🧩 Ensemble learning
🎬 Include movie title as an additional feature
👥 Include cast and crew information
📅 Add more movie metadata
🌐 Build a Streamlit web application
☁️ Deploy the model as an API
📊 Create an interactive model comparison dashboard


# **🏁 Conclusion**
This project demonstrates how NLP and Machine Learning can be used to automatically classify movies into genres.
The complete workflow is:
🧹 Preprocessing → 🔀 Train-Test Split → 📝 TF-IDF → 🤖 Model Training → 📈 Evaluation → 🎬 Prediction
Three models were compared:
🧠 Logistic Regression
⚡ Linear SVM
📊 Multinomial Naive Bayes
🏆 Logistic Regression achieved the best overall performance and was selected as the final model.
The trained model, TF-IDF vectorizer, and Label Encoder are saved as .pkl files so that predictions can be made later without retraining.

**👩‍💻 Author**
*Himangi Gupta*

# ⭐ Support
If you found this project useful:
⭐ Star the repository
🍴 Fork the repository
💬 Share feedback

# 🎬 Final Pipeline
                 🎬 MOVIE GENRE PREDICTION
                              │
                              ▼
                       📂 IMDb Dataset
                              │
                              ▼
                    🧹 Data Preprocessing
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 📅 Year          📝 Description
                    └─────────┬─────────┘
                              ▼
                       🔀 Train/Test Split
                              │
                              ▼
                         📝 TF-IDF
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
       🧠 Logistic       ⚡ Linear SVM     📊 Naive Bayes
        Regression
             └────────────────┼────────────────┘
                              ▼
                       📈 Comparison
                              │
                              ▼
                    🏆 Logistic Regression
                              │
                              ▼
                       🎬 New Prediction
                              │
                              ▼
                         🎭 Genre
