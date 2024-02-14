import clf
import pandas as pd
import tfidf
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle
import streamlit as st
import re
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

import pickle



# Download NLTK stopwords
nltk.download('stopwords')

# Define stopwords
stop_words = set(stopwords.words('english'))

def cleanResume(dfText):
    dfText = re.sub(r'https?://\S+|www\.\S+', ' ', dfText)  # remove URLs
    dfText = re.sub(r'RT|cc', ' ', dfText)  # remove RT and cc
    dfText = re.sub(r'#S+', '', dfText)  # remove hashtags
    dfText = re.sub(r'@\S+', '  ', dfText)  # remove mentions
    dfText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', dfText)  # remove punctuations
    dfText = re.sub(r'[^x00-x7f]',r' ', dfText)
    dfText = re.sub(r's+', ' ', dfText)  # remove extra whitespace

    # Convert text to lowercase
    dfText = dfText.lower()

    # Tokenize the text
    tokens = dfText.split()

    # Remove stopwords
    tokens_cleaned = [word for word in tokens if word.lower() not in stop_words]


    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens_cleaned)

    return cleaned_text

df['cleaned_resume'] = df['Resume'].apply(cleanResume)

# Encode the 'Category' column
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# TF-IDF Vectorization
stop_words = set(stopwords.words('english'))
custom_stopwords = ['in', 'on', 'of','the','for','and','a','be']
stop_words_list = list(stop_words)
tfidf_vectorizer = TfidfVectorizer(stop_words= stop_words_list+custom_stopwords, max_features=1500, sublinear_tf=True)
requiredText = df['cleaned_resume'].values
WordFeatures = tfidf_vectorizer.fit_transform(requiredText)

import pickle

# Save the TF-IDF vectorizer
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Load the TF-IDF vectorizer
with open('tfidf.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, df['Category'], test_size=0.2, random_state=0)

# Create and train the classifier
knn_classifier = KNeighborsClassifier()
clf = OneVsRestClassifier(knn_classifier)
clf.fit(X_train, y_train)



# Streamlit app
st.title("Resume Screening App")
uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

if uploaded_file is not None:
    # Read the uploaded file
    try:
        df_text = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        df_text = uploaded_file.read().decode('latin-1')


    # Define a function to check if the input text resembles a resume
    def validate_resume_text(text):
        # Define keywords commonly found in resume content
        keywords = ['experience', 'skills', 'education', 'projects', 'achievements', 'certifications', 'summary',
                    'objective']

        # Convert the input text to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Check if any of the keywords appear in the text
        if any(keyword in text_lower for keyword in keywords):
            return True
        else:
            return False


    # Check if the uploaded text resembles a resume
    if validate_resume_text(df_text):
        # Clean the resume text
        cleaned_resume = cleanResume(df_text)

        # Transform the cleaned text using TF-IDF vectorizer
        new_features = tfidf_vectorizer.transform([cleaned_resume])

        # Predict the category
        prediction = clf.predict(new_features)

        # Display the predicted category
        predicted_category = le.inverse_transform(prediction)
        st.markdown(f'<p style="color:green; font-size:20px;">Predicted Category: {predicted_category}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:red; font-size:20px;">The uploaded text does not resemble a resume.</p>', unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.title("Model Parameters")

# Add sidebar inputs for model parameters
n_neighbors = st.sidebar.slider("Number of Neighbors", min_value=1, max_value=10, value=5)

# Update the classifier with new parameter values
knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
clf = OneVsRestClassifier(knn_classifier)
clf.fit(X_train, y_train)

# Save the classifier
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

    # Load the classifier
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)







