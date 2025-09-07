# Step 1: Import libraries
import pandas as pd

# Step 2: Load the dataset
# Replace 'spam.csv' with the path to your downloaded file
df = pd.read_csv(r"C:\Users\madak\Desktop\python_libraries\scikitlearn\scikit_code\z_spamdetection\spam.csv", encoding='latin-1')

# Step 3: Inspect the dataset
print(df.head())
print(df.info())
# Drop extra columns (if any)
df = df[['v1', 'v2']]  # v1 = label, v2 = message

# Rename columns
df.columns = ['label', 'message']

# Convert labels to binary (ham=0, spam=1)
df['label'] = df['label'].map({'ham':0, 'spam':1})

# Check cleaned data
print(df.head())
print(df['label'].value_counts())

##################################################---- preprocessing ----#####################################
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
df['clean_message'] = df['message'].apply(preprocess_text)

# Check result
print(df[['message','clean_message']].head())

##################################################---- split data ----#####################################

X = df['clean_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

##################################################---- TF-IDF Vectorization ----#####################################

tfidf = TfidfVectorizer(max_features=3000)  # limit features for faster training
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

##################################################---- Model Training ----#####################################

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

# 2. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

##################################################---- Evaluation ----#####################################

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)

models = {'Naive Bayes': y_pred_nb, 'Logistic Regression': y_pred_lr, 'Random Forest': y_pred_rf}

for name, y_pred in models.items():
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")
