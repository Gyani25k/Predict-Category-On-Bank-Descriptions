import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from collections import Counter



df = pd.read_csv('cleaned_data.csv')

try:
    df = df.drop(columns=['Unnamed: 0'])
except:
    df = df

df = df.dropna()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=100)
description_tfidf = tfidf.fit_transform(df['Description'])

# One-Hot Encoding for 'Type'
encoder = OneHotEncoder(sparse_output=False)
type_encoded = encoder.fit_transform(df[['Type']])

# Combine features
X = np.hstack((description_tfidf.toarray(), type_encoded))

# Target variable
y = df['Category']

# Check class distribution before resampling
class_counts = Counter(y)
print("Class distribution before resampling:", class_counts)

# Apply SMOTE only if there are enough samples for minority classes
if all(count >= 6 for count in class_counts.values()):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    print("Not enough samples for minority classes to apply SMOTE. Skipping resampling.")
    X_resampled, y_resampled = X, y

# Check the distribution after resampling
if isinstance(y_resampled, np.ndarray):
    print("Class distribution after resampling:", Counter(y_resampled))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

with open('model_results.txt','w') as file:
    file.write("Classification Report \n\n")
    file.write(classification_report(y_test, y_pred))
    file.write("\n\n")
    file.write("Confusion Matrix \n\n")
    cm_str = np.array2string(confusion_matrix(y_test, y_pred))
    file.write(cm_str)

# Distribution of Categories
plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=df)
plt.title('Distribution of Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.savefig('Distribution_of_Categories.png')

# Word Cloud of Descriptions
text = ' '.join(df['Description'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Descriptions')
plt.savefig('WordCloud.png')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Confusion_Matrix.png')

# Save the model, TF-IDF vectorizer, and OneHotEncoder using pickle
with open('./model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('./model/onehot_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('./model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)


