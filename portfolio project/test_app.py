import pytest
from flask import url_for
from app import app, clean_text  # Replace with the actual filename of your Flask app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get(url_for('home'))
    assert response.status_code == 200
    assert b'index.html' in response.data  # Ensure template is used

def test_predict_route_with_valid_data(client):
    form_data = {
        'title': 'Sample Project Title',
        'essay1': 'This is a sample essay 1.',
        'essay2': 'This is a sample essay 2.',
        'summary': 'This is a sample resource summary.',
        'state': 'CA',
        'grade': 'Grades 3-5',
        'subject': 'Math & Science',
        'quantity': '10',
        'price': '100.0',
        'previous_projects': '5'
    }
    response = client.post(url_for('predict'), data=form_data)
    assert response.status_code == 200
    assert b'index.html' in response.data
    assert b'Approved' in response.data or b'Not Approved' in response.data

def test_predict_route_with_invalid_data(client):
    form_data = {
        'title': 'Sample Project Title',
        'essay1': 'This is a sample essay 1.',
        'essay2': 'This is a sample essay 2.',
        'summary': 'This is a sample resource summary.',
        'state': 'CA',
        'grade': 'Grades 3-5',
        'subject': 'Math & Science',
    }
    response = client.post(url_for('predict'), data=form_data)
    assert response.status_code == 200
    assert b'index.html' in response.data
    assert b'Error' in response.data

def test_predict_route_with_invalid_numerical_data(client):
    form_data = {
        'title': 'Sample Project Title',
        'essay1': 'This is a sample essay 1.',
        'essay2': 'This is a sample essay 2.',
        'summary': 'This is a sample resource summary.',
        'state': 'CA',
        'grade': 'Grades 3-5',
        'subject': 'Math & Science',
        'quantity': 'invalid',
        'price': 'invalid',
        'previous_projects': 'invalid'
    }
    response = client.post(url_for('predict'), data=form_data)
    assert response.status_code == 200
    assert b'index.html' in response.data
    assert b'Error' in response.data

def test_clean_text_function():
    test_text = "This is a sample text with some stopwords like 'the', 'and', 'is'."
    cleaned_text = clean_text(test_text)
    assert 'the' not in cleaned_text
    assert 'and' not in cleaned_text
    assert 'is' not in cleaned_text
    assert cleaned_text == 'sample text stopwords like'
