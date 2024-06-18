import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
import joblib



@st.cache_resource
def load_model():
    # Load TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Load SVM classifier
    svm_classifier = joblib.load('svm_model.pkl')

    return tfidf_vectorizer, svm_classifier

def predict_language(text, tfidf_vectorizer, svm_model):
    # Preprocess input text (e.g., tokenization, TF-IDF transformation)
    text_tfidf = tfidf_vectorizer.transform([text])

    # Make prediction using the SVM model
    prediction = svm_model.predict(text_tfidf)
    return prediction[0]


def main():

    # Load model
    tfidf_vectorizer, svm_classifier = load_model()

    st.title("Language Identifier")

    user_input = st.text_area("Enter text for language identification:")

    if st.button("Identify Language"):
        # Predict language using the SVM model
        if user_input:
            language_prediction = predict_language(user_input, tfidf_vectorizer, svm_classifier)
            st.write("Predicted Language:", language_prediction)
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
