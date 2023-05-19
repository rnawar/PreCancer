import pytest
from flask import Flask
from flask_testing import TestCase
from io import BytesIO

from main import *


class MyAppTestCase(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_home_page(self):
        response = self.client.get('/')
        self.assert200(response)
        self.assert_template_used('home.html')

    def test_prediction_page(self):
        response = self.client.get('/prediction')
        self.assert200(response)
        self.assert_template_used('prediction.html')

    def test_faq_page(self):
        response = self.client.get('/faq')
        self.assert200(response)
        self.assert_template_used('faq.html')

    def test_predict_using_model_page(self):
        response = self.client.get('/predict_using_model')
        self.assert200(response)
        self.assert_template_used('predict_using_model.html')


    def test_trained_model_result_page_with_valid_csv_file(self):
        data = {'myfile': (BytesIO(b'col1,col2\n1,2\n3,4\n'), 'test.csv')}
        response = self.client.post('/trained_model_result', data=data, content_type='multipart/form-data')
        self.assert200(response)
        self.assert_template_used('trained_model_result.html')


    def test_download_results(self):
        response = self.client.get('/download_results')
        self.assert200(response)
        self.assertEqual(response.content_type, 'text/csv; charset=utf-8')
        self.assertEqual(response.headers['Content-Disposition'], 'attachment; filename=preprocessed_dataset.csv')



if __name__ == '__main__':
    pytest.main()
