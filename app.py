import os
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename
#import keras
import keras.preprocessing as kp
import pandas as pd
import numpy as np
import pickle



UPLOAD_FOLDER = './testing'
ALLOWED_EXTENSIONS = set(['png', 'PNG'])

# create app passing the __name__
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#allowed file types only
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #upload
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #test & results
            filename = "./models/face_model"
            model = pickle.load(open(filename, "rb"))
            path = "testing/"
            filelist = os.listdir(path)
            for x in filelist:
                if x.endswith(".png"):
                    img = kp.image.load_img(path+x, target_size=(227, 227))
                    y = kp.image.img_to_array(img)
                    y = y.transpose(2,0,1).reshape(1,-1)
                    guess = model.predict(y)
                    if guess[0] == 1:
                        print('real')
                        return render_template("indexr.html")
                    else:
                        print('fake')
                        return render_template("indexf.html")
                if x.endswith(".PNG"):
                    img = kp.image.load_img(path+x, target_size=(227, 227))
                    y = kp.image.img_to_array(img)
                    y = y.transpose(2,0,1).reshape(1,-1)
                    if guess[0] == 1:
                        print('real')
                        return render_template("indexr.html")
                    else:
                        print('fake')
                        return render_template("indexf.html")
    return render_template("index.html")

@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
   app.run(debug = True)
