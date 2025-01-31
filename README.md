# Project Approval Predictor

## Overview
The **Project Approval Predictor** is a Flask-based web application that predicts whether a project will be approved or not based on various features such as project title, essays, resource summary, school state, grade category, subject category, and more. It also provides the probability of approval and non-approval.

## Features
- User-friendly web interface for project submission.
- Predicts whether a project will be approved.
- Displays probabilities for both approval and non-approval.
- Color-coded results: Green for "Approved" and Red for "Not Approved".
- Uses **TF-IDF**, **One-Hot Encoding**, **PCA**, and an **XGBoost Model** for prediction.

## Tech Stack
- **Backend**: Flask (Python)
- **Machine Learning Model**: XGBoost
- **Frontend**: HTML, CSS 
- **Data Processing**: Pandas, Scikit-learn

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/yuvrajsinghchauhan01/Portfolio-Project.git
cd portfolio project
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```

### 5. Open in Browser
Visit: `http://127.0.0.1:5000/`

## Usage
1. Enter the project details in the web form.
2. Click the **Predict Approval** button.
3. View the prediction result along with probabilities.

## Expected Output
- **Approved (Green)**: If the project is likely to be approved.
- **Not Approved (Red)**: If the project is unlikely to be approved.
- **Probability Display**: Shows confidence for both classes.

## Model Workflow
1. **Text Processing**: TF-IDF transformation of text fields (Project Title, Essays, Resource Summary).
2. **Categorical Encoding**: One-Hot Encoding for categorical features.
3. **Dimensionality Reduction**: PCA is applied to reduce feature space.
4. **Prediction**: XGBoost model predicts approval status.

## File Structure
```
portfolio_project/
│── __pycache__/
│── Dataset/
│── notebook/
│── templates/
│── .gitignore
│── app.py
│── Dockerfile
│── feature_columns.pkl
│── flask_test.py
│── LICENSE
│── model.py
│── ohe_encoder.pkl
│── pca.pkl
│── README.md
│── requirements.txt
│── scaler.pkl
│── tfidf_transformer.pkl
│── xgb_model.pkl
```

## Future Improvements
- Allow batch predictions.
- Improve UI with better design and responsiveness.
- Implement additional ML models for better accuracy.

## Contributing
Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License.

---
### 🚀 Happy Coding! 🎯

