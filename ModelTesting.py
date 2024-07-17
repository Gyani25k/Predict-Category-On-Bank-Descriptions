import pandas as pd
import numpy as np
import nltk
import pickle

nltk.download('stopwords')
nltk.download('punkt')

# Load the saved models and encoders
with open('./model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('./model/onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('./model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

new_df = pd.read_csv('cleaned_data.csv')

try:
    new_df = new_df.drop(columns=['Unnamed: 0'])
except:
    new_df = new_df

new_df = new_df.dropna()

# Apply preprocessing to new data
new_df['Description'] = new_df['Description'].apply(preprocess_text)

# Transform new data
new_description_tfidf = tfidf.transform(new_df['Description'])
new_type_encoded = encoder.transform(new_df[['Type']])
new_X = np.hstack((new_description_tfidf.toarray(), new_type_encoded))

# Predict
new_pred = model.predict(new_X)

# Apply rules
def apply_rules(description, transaction_type, predicted_category):
    if transaction_type == 'CR' and predicted_category in ['Opex', 'Loan Repayment']:
        return None  # Invalid category for Credit
    if transaction_type == 'DB' and predicted_category in ['Revenue', 'Loan Received']:
        return None  # Invalid category for Debit
    return predicted_category

new_df['Predicted_Category'] = [apply_rules(desc, t_type, pred) for desc, t_type, pred in zip(new_df['Description'], new_df['Type'], new_pred)]

new_df.to_csv('Result_Prediction.csv')