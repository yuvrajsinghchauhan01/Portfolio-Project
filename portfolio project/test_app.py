import unittest
from flask import Flask, url_for
from flask_testing import TestCase
from app import app  # Replace with the actual filename of your Flask app

class TestFlaskApp(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        return app

    def test_home_route(self):
        response = self.client.get(url_for('home'))
        self.assert200(response)
        self.assertTemplateUsed('index.html')

    def test_predict_route_with_valid_data(self):
        # Mock form data
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

        response = self.client.post(url_for('predict'), data=form_data)
        self.assert200(response)
        self.assertTemplateUsed('index.html')
        self.assertIn(b'Approved', response.data)  # or 'Not Approved' depending on the model's prediction

    def test_predict_route_with_invalid_data(self):
        # Mock form data with missing required fields
        form_data = {
            'title': 'Sample Project Title',
            'essay1': 'This is a sample essay 1.',
            'essay2': 'This is a sample essay 2.',
            'summary': 'This is a sample resource summary.',
            'state': 'CA',
            'grade': 'Grades 3-5',
            'subject': 'Math & Science',
            # Missing 'quantity', 'price', and 'previous_projects'
        }

        response = self.client.post(url_for('predict'), data=form_data)
        self.assert200(response)
        self.assertTemplateUsed('index.html')
        self.assertIn(b'Error', response.data)  # Expecting an error message

    def test_predict_route_with_invalid_numerical_data(self):
        # Mock form data with invalid numerical values
        form_data = {
            'title': 'Sample Project Title',
            'essay1': 'This is a sample essay 1.',
            'essay2': 'This is a sample essay 2.',
            'summary': 'This is a sample resource summary.',
            'state': 'CA',
            'grade': 'Grades 3-5',
            'subject': 'Math & Science',
            'quantity': 'invalid',  # Invalid quantity
            'price': 'invalid',     # Invalid price
            'previous_projects': 'invalid'  # Invalid previous projects
        }

        response = self.client.post(url_for('predict'), data=form_data)
        self.assert200(response)
        self.assertTemplateUsed('index.html')
        self.assertIn(b'Error', response.data)  # Expecting an error message

    def test_clean_text_function(self):
        from app import clean_text  # Replace with the actual filename of your Flask app

        test_text = "This is a sample text with some stopwords like 'the', 'and', 'is'."
        cleaned_text = clean_text(test_text)
        self.assertNotIn('the', cleaned_text)
        self.assertNotIn('and', cleaned_text)
        self.assertNotIn('is', cleaned_text)
        self.assertEqual(cleaned_text, 'sample text stopwords like')

if __name__ == '__main__':
    unittest.main()