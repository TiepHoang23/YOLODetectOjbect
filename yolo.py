
from logging.config import dictConfig
from re import I
from xml.parsers.expat import model
import numpy as np
import cv2

IOU_THRESHOLD = 0.5


class YOLO:
    # init configuration and weight file for model and init network
    def __init__(self, labels_file, config_file, weights_file, confidence_threshold):

        self.labels = open(labels_file).read().strip().split("\n")
        self.CONFIDENCE_THRESHOLD = float(confidence_threshold)
        self.net_model = cv2.dnn.readNetFromDarknet(
            config_file, weights_file)
    # Do dectect bouding and get confidences, ClassID

    def detectData(self, layerOutputs, image):

        boxes = []
        confidences = []
        classIDs = []
        (H, W) = image.shape[:2]  # image height and weight
        for output in layerOutputs:

            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.CONFIDENCE_THRESHOLD:
                    # scale the bounding box relative to the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # get index top-left
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences, ClassID
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        return boxes, confidences, classIDs
    # Set bounding box

    def DrawBoxes(self, boxes, labels, confidences, classIDs, image):
        # Do Non-max suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD,
                                IOU_THRESHOLD)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # Set bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (255, 0, 0), 2)
                text = "{}: {:.4f}".format(
                    labels[classIDs[i]], confidences[i])
                # Set text confident
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)
    # Do update config ,weight for model and update network

    def updateModel(self, labels_file, config_file, weights_file, confidence_threshold):
        self.labels = open(labels_file).read().strip().split("\n")
        self.CONFIDENCE_THRESHOLD = float(confidence_threshold)
        self.net_model = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    # Detect image and find object

    def detectObjects(self, input_file):
        # determine only the *output* layer names that we need from YOLO
        image = cv2.imread(input_file)
        ln = self.net_model.getLayerNames()
        ln = [ln[i - 1] for i in self.net_model.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)

        self.net_model.setInput(blob)
        layerOutputs = self.net_model.forward(ln)
        # pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
        boxes, confidences, classIDs = self.detectData(layerOutputs, image)

        self.DrawBoxes(boxes, self.labels, confidences, classIDs, image)
        return image


modelName = ["model01.txt", "model02.txt"]
curmodel = modelName[0][:-4]

# Read model in file models and get cogfig with state CONFIG_NAME: path/


def getConfigModel():
    list_config_models = dict()
    for name in modelName:
        Config = dict()
        path = 'models/' + name
        listCfg = open(path, 'r').read().strip().split("\n")
        fileConfigs = list(map(lambda x: x.split(" = "), listCfg))
        for cfg in fileConfigs:
            Config[cfg[0]] = cfg[1]
        list_config_models[name] = Config
    return list_config_models


Configs = getConfigModel()

# init model with config file


def initModel():
    Config = Configs[modelName[0]]
    return YOLO(Config["LABELS_FILE"],
                Config["CONFIG_FILE"],
                Config["WEIGHTS_FILE"],
                Config["CONFIDENCE_THRESHOLD"])

# update model with new config file


def updateModel(model, name):
    Config = Configs[name]
    model.updateModel(Config["LABELS_FILE"],
                      Config["CONFIG_FILE"],
                      Config["WEIGHTS_FILE"],
                      Config["CONFIDENCE_THRESHOLD"])
