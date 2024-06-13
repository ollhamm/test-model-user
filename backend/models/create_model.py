import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error

# Load datasets
data = pd.read_csv('/content/drive/MyDrive/data-rekomendasi/setelah_cleaning.csv')

# Process data
data['Category'] = data['Category'].fillna('')

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Category'])

# Convert TF-IDF matrix to dense NumPy array
tfidf_matrix_dense = tfidf_matrix.toarray()

def train_model():
    # Define the recommendation model
    input_layer = Input(shape=(tfidf_matrix_dense.shape[1],), name='input')
    encoded = Dense(100, activation='relu')(input_layer)
    decoded = Dense(tfidf_matrix_dense.shape[1], activation='softmax')(encoded)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')  

    autoencoder.fit(tfidf_matrix_dense, tfidf_matrix_dense, epochs=10, batch_size=32)

    return autoencoder  

if __name__ == '__main__':
    model = train_model()
    model.save('models_category_user.h5')
