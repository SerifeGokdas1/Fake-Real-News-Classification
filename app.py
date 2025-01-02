import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import torch
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from transformers import AutoTokenizer


# Tokenizer'ı başlatıyoruz
tokenizer = AutoTokenizer.from_pretrained("alibayram/tr_tokenizer", use_fast=True)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Model dosyaları
log_reg_model = joblib.load('logistic_regression_model.pkl')  # Logistic Regression modeli
#lstm_model = tf.keras.models.load_model('lstm_model.h5')  # LSTM model
nn_model = joblib.load('neural_network_model.pkl')  # Neural network modeli
rf_model = joblib.load('random_forest_model.pkl')  # Random Forest modeli
svm_model = joblib.load('support_vector_machine_model.pkl')  # SVM modeli
xgboost_model = joblib.load('xgboost_model.pkl')  # XGBoost modeli



# Preprocessing ve veri hazırlığı için fonksiyonlar
def preprocess_text(text):
    STOPWORDS = set(stopwords.words('turkish'))
    STOPWORDS.add('mi') 
    # Küçük harfe çevirme
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    # Stopwords kaldır ve lemmatization uygula
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def tokenize_text(text):
    
    tokens = tokenizer.tokenize(text)
    return tokens

def vectorize_text(text):
    # Tokenized listeyi string'e dönüştürelim
    titles_as_strings = [' '.join(eval(tokens)) for tokens in text]
    # Metni vektöre dönüştürme
    return embedding_model.encode(titles_as_strings)

def vectorize_text(text):
    
    # Her bir başlığı string olarak kabul ederek işleme alıyoruz
    titles_as_strings = [' '.join(tokens) if isinstance(tokens, list) else tokens for tokens in text]
    
    # Metni vektöre dönüştürme
    return embedding_model.encode(titles_as_strings)



# Tahmin fonksiyonları
def predict_logistic_regression(text):
    
    return log_reg_model.predict(text)



def predict_neural_network(text):

    return nn_model.predict(text)

def predict_random_forest(text):
    
    return rf_model.predict(text)

def predict_svm(text):
    
    return svm_model.predict(text)

def predict_xgboost(text):
    
    return xgboost_model.predict(text)

# Streamlit arayüzü
st.title("Haber Başlığı Doğruluk Tahmin Aracı")

# Kullanıcıdan haber başlığını alma
news_title = st.text_input("Haber Başlığını Girin:")

title_preprocesed=preprocess_text(news_title)
title_tokenized=tokenize_text(title_preprocesed)
title_vectorized=vectorize_text(title_tokenized)

# Model Seçim kutusu
model_choice = st.selectbox(
    "Hangi modeli kullanmak istersiniz?",
    ("Logistic Regression", "LSTM","Neural Network", "Random Forest", "Support Vector Machine", "XGBoost")
)

# "Tahmin Et" butonu
if st.button("Tahmin Et"):
    if news_title:
        # Seçilen modele göre tahmin yapma
        if model_choice == "Logistic Regression":
            prediction = predict_logistic_regression(title_vectorized)
        elif model_choice == "Neural Network":
            prediction = predict_neural_network(title_vectorized)
        elif model_choice == "Random Forest":
            prediction = predict_random_forest(title_vectorized)
        elif model_choice == "Support Vector Machine":
            prediction = predict_svm(title_vectorized)
        elif model_choice == "XGBoost":
            prediction = predict_xgboost(title_vectorized)



        # Tahmin Sonucunu Gösterme
        if prediction[0] == 1:
            st.write(f"**Tahmin:** Bu haber başlığı doğru.")
        else:
            st.write(f"**Tahmin:** Bu haber başlığı yanlış.")
        
        # Tüm modellerin tahminlerini yazma
        st.write("### Tüm Modellerin Tahminleri:")
        st.write(f"**Logistic Regression Tahmini:** {'Doğru' if predict_logistic_regression(title_vectorized) == 1 else 'Yanlış'}")
        st.write(f"**Neural Network Tahmini:** {'Doğru' if predict_neural_network(title_vectorized) == 1 else 'Yanlış'}")
        st.write(f"**Random Forest Tahmini:** {'Doğru' if predict_random_forest(title_vectorized) == 1 else 'Yanlış'}")
        st.write(f"**SVM Tahmini:** {'Doğru' if predict_svm(title_vectorized) == 1 else 'Yanlış'}")
        st.write(f"**XGBoost Tahmini:** {'Doğru' if predict_xgboost(title_vectorized) == 1 else 'Yanlış'}")
    else:
        st.write("Lütfen bir haber başlığı girin.")
