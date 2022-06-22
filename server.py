import yolo
import os
from flask import Flask, redirect, render_template, request, session
from werkzeug.utils import secure_filename
import cv2
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# init model when start server
# model default is model01
model_yolo = yolo.initModel()

# Route home


@app.route("/")
def uploader():
    path = 'static/uploads/'
    uploads = sorted(os.listdir(
        path), key=lambda x: os.path.getctime(path + x))
    uploads = ['uploads/' + file
               for file in uploads]
    modelName = [item[:-4] for item in yolo.modelName]

    return render_template("index.html", uploads=uploads, modelName=modelName, curModel=yolo.curmodel)


app.config['UPLOAD_PATH'] = 'static/uploads'

# Route update model


@app.route("/update", methods=['GET', 'POST'])
def updateModel():
    if request.method == 'POST':
        idx = request.form.get("model")
        # When user want update model, action update
        yolo.curmodel = yolo.modelName[int(idx)][:-4]
        yolo.updateModel(model_yolo,  yolo.modelName[int(idx)])
    return redirect("/")


# Route upload file

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    dir = 'static/uploads'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        # Path Input file
        INPUT_FILE = 'static/uploads/' + filename
        # Path ouput file
        OUTPUT_FILE = filename.split(
            '.')[0] + "_predict." + filename.split('.')[1]
        # Detect image
        image = model_yolo.detectObjects(INPUT_FILE)
        cv2.imwrite("static/uploads/{}".format(OUTPUT_FILE), image)

        return redirect("/")


if __name__ == "__main__":

    app.run()
