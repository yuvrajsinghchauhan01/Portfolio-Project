from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load saved components
with open('tfidf_transformers.pkl', 'rb') as f:
    tfidf_transformers = pickle.load(f)
with open('ohe_encoder.pkl', 'rb') as f:
    ohe_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Get unique values for dropdowns
school_states = sorted(['IN', 'FL', 'AZ', 'KY', 'TX', 'CT', 'GA', 'SC', 'NC', 'CA', 'NY',
       'OK', 'MA', 'NV', 'OH', 'PA', 'AL', 'LA', 'VA', 'AR', 'WA', 'WV',
       'ID', 'TN', 'MS', 'CO', 'UT', 'IL', 'MI', 'HI', 'IA', 'RI', 'NJ',
       'MO', 'DE', 'MN', 'ME', 'WY', 'ND', 'OR', 'AK', 'MD', 'WI', 'SD',
       'NE', 'NM', 'DC', 'KS', 'MT', 'NH', 'VT'])
grade_categories = sorted(['Grades PreK-2', 'Grades 6-8', 'Grades 3-5', 'Grades 9-12'])
subject_categories = sorted(['Literacy & Language', 'History & Civics, Health & Sports',
       'Health & Sports', 'Literacy & Language, Math & Science',
       'Math & Science', 'Literacy & Language, Special Needs',
       'Literacy & Language, Applied Learning', 'Special Needs',
       'Math & Science, Literacy & Language', 'Applied Learning',
       'Math & Science, Special Needs', 'Music & The Arts',
       'History & Civics', 'Health & Sports, Literacy & Language',
       'Literacy & Language, Music & The Arts', 'Warmth, Care & Hunger',
       'Math & Science, History & Civics',
       'Applied Learning, Literacy & Language',
       'Applied Learning, Special Needs',
       'Literacy & Language, History & Civics',
       'Applied Learning, Health & Sports',
       'History & Civics, Literacy & Language',
       'Health & Sports, Special Needs',
       'Applied Learning, Math & Science',
       'Math & Science, Music & The Arts',
       'Health & Sports, Applied Learning',
       'History & Civics, Music & The Arts',
       'Math & Science, Applied Learning',
       'Music & The Arts, History & Civics',
       'Applied Learning, Music & The Arts',
       'History & Civics, Math & Science',
       'Music & The Arts, Applied Learning',
       'Health & Sports, Music & The Arts',
       'Math & Science, Health & Sports',
       'Special Needs, Health & Sports',
       'Health & Sports, Math & Science',
       'Special Needs, Music & The Arts',
       'Music & The Arts, Warmth, Care & Hunger',
       'Applied Learning, History & Civics',
       'Music & The Arts, Special Needs',
       'Health & Sports, History & Civics',
       'History & Civics, Applied Learning',
       'Literacy & Language, Warmth, Care & Hunger',
       'History & Civics, Special Needs',
       'Health & Sports, Warmth, Care & Hunger',
       'Music & The Arts, Health & Sports',
       'Applied Learning, Warmth, Care & Hunger',
       'Literacy & Language, Health & Sports',
       'Math & Science, Warmth, Care & Hunger',
       'Special Needs, Warmth, Care & Hunger',
       'History & Civics, Warmth, Care & Hunger'])

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html', 
                           school_states=school_states,
                           grade_categories=grade_categories,
                           subject_categories=subject_categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'project_title': request.form['title'],
            'project_essay_1': request.form['essay1'],
            'project_essay_2': request.form['essay2'],
            'project_resource_summary': request.form['summary'],
            'school_state': request.form['state'],
            'project_grade_category': request.form['grade'],
            'project_subject_categories': request.form['subject'],
            'quantity': float(request.form['quantity']),
            'price': float(request.form['price']),
            'teacher_number_of_previously_posted_projects': int(request.form['previous_projects'])
        }
        
        input_df = pd.DataFrame([data])
        
        # Process text columns
        text_features = []
        for col in tfidf_transformers.keys():
            cleaned_text = input_df[col].fillna('').apply(clean_text)
            tfidf = tfidf_transformers[col]
            tfidf_matrix = tfidf.transform(cleaned_text)
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                                  columns=[f"{col}_{word}" for word in tfidf.get_feature_names_out()])
            text_features.append(tfidf_df)
        
        # Process categorical columns
        categorical_data = input_df[ohe_encoder.feature_names_in_]
        ohe_matrix = ohe_encoder.transform(categorical_data)
        ohe_df = pd.DataFrame(ohe_matrix, columns=ohe_encoder.get_feature_names_out())
        
        # Combine all features
        numerical_df = input_df[['quantity', 'price', 'teacher_number_of_previously_posted_projects']]
        processed_df = pd.concat([numerical_df, pd.concat(text_features, axis=1), ohe_df], axis=1)
        
        # Align columns with training data
        processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)
        
        # Scale and PCA
        scaled_data = scaler.transform(processed_df)
        pca_data = pca.transform(scaled_data)
        
        # Predict
        prediction = model.predict(pca_data)
        result = "Approved" if prediction[0] == 1 else "Not Approved"
        
        # Get probability percentages
        probabilities = model.predict_proba(pca_data)[0]
        probability_approved = round(probabilities[1] * 100, 2)  # Probability of "Approved"
        probability_not_approved = round(probabilities[0] * 100, 2)  # Probability of "Not Approved"
        
        return render_template('index.html', 
                              prediction=result,
                              probability_approved=probability_approved,
                              probability_not_approved=probability_not_approved,
                              school_states=school_states,
                              grade_categories=grade_categories,
                              subject_categories=subject_categories)
    
    except Exception as e:
        return render_template('index.html', 
                              prediction=f"Error: {str(e)}",
                              school_states=school_states,
                              grade_categories=grade_categories,
                              subject_categories=subject_categories)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)