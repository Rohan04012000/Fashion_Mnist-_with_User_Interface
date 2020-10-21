from flask import Flask, request, url_for, redirect, render_template
import numpy as np
import pandas as pd
from keras.models import load_model

app = Flask(__name__)
model = load_model('keras_model.h5')
testing = pd.read_csv(r"fashion-mnist_test.csv")

@app.route('/')
def hello_world():
    return render_template("webpage.html")

@app.route('/predict', methods = ['POST','GET'])
def predict():
    item_number = 0
    for x in request.form.values():
        item_number = int(x)
        break

    x = testing.iloc[item_number]
    y = x['label']
    z = x.drop(labels = "label")
    z = np.array([z]) / 255
    a1 = model.predict_classes(z)
    predict = a1[0]

    if predict == 0:
        return render_template('webpage.html', pred =' Answer predicted by model is == T-shirt/Top')
    if predict == 1:
        return render_template('webpage.html', pred = ' Answer predicted by model is == Trouser')
    if predict == 2:
        return render_template('webpage.html', pred = ' Answer predicted by model is == Pullover')
    if predict == 3:
        return render_template('webpage.html', pred = ' Answer predicted by model is == Dress')
    if predict == 4:
        return render_template('webpage.html', pred = ' Answer predicted by model is == Coat')
    if predict == 5:
        return render_template('webpage.html', pred = ' Answer predicted by model is == Sandal')
    if predict == 6:
        return render_template('webpage.html', pred = ' Answer predicted by model is == Shirt')
    if predict == 7:
        return render_template('webpage.html', pred = ' Answer predicted by model is == Sneaker')
    if predict == 8:
        return render_template('webpage.html', pred = 'Answer predicted by model is == Bag')
    if predict == 9:
        return render_template('webpage.html', pred = 'Answer predicted by model is == Ankle Boot')

if __name__ == '__main__':
    app.run(debug=True)