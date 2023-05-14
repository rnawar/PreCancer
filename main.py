from flask import Flask, render_template, request, redirect, flash, url_for, Response
from werkzeug.utils import secure_filename
import pandas as pd
from model import *

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)

my_bagging = Bagging()


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', home=True)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html', prediction=True)


@app.route('/faq')
def faq():
    return render_template('faq.html', faq=True)


@app.route('/predict_using_model')
def predict_using_model():
    return render_template('predict_using_model.html', predict_using_model=True)


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        f = request.files['myfile']

        if f.filename == '':
            flash('Please Upload a File', 'error')
            return redirect(request.url)

        if f and allowed_file(f.filename):
            res = pd.read_csv(f, low_memory=False)
            # get the column names
            column_names = list(res.columns)
            # check if the target column name is valid
            target_col_name = request.form.get('myTargetColumn')
            if target_col_name not in column_names:
                flash('Invalid target column name', 'error')
                return redirect(request.url)
            number_of_column = int(request.form.get('myNumberOfColumn'))
            if number_of_column <= 0:
                flash('Please enter a valid number of columns greater than 0.', 'error')
                return redirect(request.url)
            elif number_of_column > len(res.columns) - 2:  # - 2 since we need to exclude the first and last column
                flash('The number of columns cannot be greater than the number of columns in the dataset.', 'error')
                return redirect(request.url)
            result = my_bagging.algorithm(res, target_col_name, number_of_column)
            return render_template('result.html',
                                   data={'accuracy': result['accuracy'], 'sensitivity': result['sensitivity'],
                                         'specificity': result['specificity'], 'auc': result['auc']})
        else:
            flash('Wrong File Format', 'error')

    return redirect(url_for('prediction'))


@app.route('/trained_model_result', methods=['GET', 'POST'])
def trained_model_result():
    if request.method == 'POST':
        f = request.files['myfile']

        if f.filename == '':
            flash('Please Upload a File', 'error')
            return redirect(request.url)

        if f and allowed_file(f.filename):
            res = pd.read_csv(f, low_memory=False)
            my_bagging.set_new_data(res)
            return render_template('trained_model_result.html', trained_model_result=True)
        else:
            flash('Wrong File Format', 'error')

    return redirect(url_for('predict_using_model'))


@app.route('/download_results')
def download_results():
    # Get the results data from the session or database
    csv_data = my_bagging.save_preprocessed_dataset()

    # Convert DataFrame to CSV string
    csv_string = csv_data.to_csv(index=False)

    # Return the CSV file as a file download
    return Response(
        csv_string,
        mimetype="text/csv",
        headers={"Content-disposition":
                     "attachment; filename=preprocessed_dataset.csv"})


@app.route('/download_new_results')
def download_new_results():
    # Get the results data from the session or database
    csv_data = my_bagging.predict_new_data()

    # Convert DataFrame to CSV string
    csv_string = csv_data.to_csv(index=False)

    # Return the CSV file as a file download
    return Response(
        csv_string,
        mimetype="text/csv",
        headers={"Content-disposition":
                     "attachment; filename=predicted_class_label.csv"})


if __name__ == "__main__":
    app.secret_key = 'Hello World'
    app.run(debug=True)
