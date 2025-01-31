# Import Libraries
import pandas as pd
import numpy as np
import re
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix
)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
import nltk
nltk.download('punkt_tab')

# Load Dataset
train_data = pd.read_csv("/Users/yuvraj/Desktop/portfolio-project/portfolio project/Dataset/train_data.csv")
resource_data = pd.read_csv("/Users/yuvraj/Desktop/portfolio-project/portfolio project/Dataset/resources.csv")

# Merge Datasets
merge_df = pd.merge(train_data, resource_data, on='id', how='inner')

# Drop Columns
merge_df.drop(columns=['Unnamed: 0', 'id', 'teacher_id',
'project_submitted_datetime', 'project_essay_3', 'project_essay_4',
'project_subject_subcategories','description','teacher_prefix'],inplace=True)

# Clean Text Function and Tfidf Vectorizer
def clean_text(text):
    """
    Cleans the text by removing punctuation, converting to lowercase,
    tokenizing, and removing stopwords.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    stop_words = set(stopwords.words('english'))
    # Remove punctuation and symbols
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Preprocess and Save TF-IDF (modified to return transformers)
def preprocess_and_save_tfidf(df, text_columns, max_features=None):
    tfidf_features_list = []
    tfidf_transformers = {}

    for col in text_columns:
        df[col] = df[col].fillna('').apply(clean_text)
        tfidf = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = tfidf.fit_transform(df[col])
        tfidf_transformers[col] = tfidf
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=[f"{col}_{word}" for word in tfidf.get_feature_names_out()])
        tfidf_features_list.append(tfidf_df)
    
    combined_tfidf = pd.concat(tfidf_features_list, axis=1)
    return combined_tfidf, tfidf_transformers




# One-hot Encoding with OneHotEncoder (modified)
def one_hot_encode_and_save(df, categorical_columns):
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        one_hot_encoded = ohe.fit_transform(df[categorical_columns])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, 
                                         columns=ohe.get_feature_names_out(categorical_columns))
        return one_hot_encoded_df, ohe
    except Exception as e:
        print(f"Error during one-hot encoding: {e}")
        return pd.DataFrame(), None

# Preprocess text columns and get transformers
text_columns = ['project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary']
tfidf_features, tfidf_transformers = preprocess_and_save_tfidf(merge_df, text_columns, max_features=50)
print('tfidf--done')
# One-hot encode categorical columns
categorical_columns = ['school_state','project_grade_category','project_subject_categories']
one_hot_encoded_df, ohe_encoder = one_hot_encode_and_save(merge_df, categorical_columns)

# Prepare final DataFrame (same as original)
df = merge_df.copy()
df.drop(columns=['school_state', 'project_grade_category', 'project_subject_categories',
       'project_title', 'project_essay_1', 'project_essay_2',
       'project_resource_summary'], axis=1, inplace=True)
final_df = pd.concat([df, tfidf_features, one_hot_encoded_df], axis=1)

# Save feature column order
X = final_df.drop(columns='project_is_approved', axis=1)
feature_columns = X.columns.tolist()
y = final_df['project_is_approved']

# Train-test split (same as original)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE, scaling, and PCA (same as original)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_resampled_pca = pca.fit_transform(X_resampled_scaled)
X_test_pca = pca.transform(X_test_scaled)
print('pca--done')
# Train model (same as original)
xgb = XGBClassifier(n_estimators=100, max_depth=10, n_jobs=-1, 
                     eval_metric='logloss', random_state=42)
xgb.fit(X_resampled_pca, y_resampled)

# Predict on test set
y_pred = xgb.predict(X_test_pca)
y_pred_proba = xgb.predict_proba(X_test_pca)[:, 1]

# Evaluate model (same as original)
accuracy = accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", confusion)

# Save all required components
with open('tfidf_transformers.pkl', 'wb') as f:
    pickle.dump(tfidf_transformers, f)
with open('ohe_encoder.pkl', 'wb') as f:
    pickle.dump(ohe_encoder, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)