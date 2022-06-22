import cv2
import os
import yolo
import numpy

model_yolo = yolo.initModel()
yolo.updateModel(model_yolo, "model02.txt")
path1 = "test/img/"
path2 = "test/label/"
images = sorted(os.listdir(path1))
# print(images)

for image in images:

    predict_img = model_yolo.detectObjects(
        path1+image)
    cv2.imwrite("test/predict/"+image, predict_img)
    # print(predict_img)
    pass
