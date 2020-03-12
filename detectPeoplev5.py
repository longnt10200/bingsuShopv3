import cv2 as cv
import numpy as np 
import os 
from sort2 import Sort
import IrregularBox as irbox 
import datetime
import time

def load_model_keras(modelname):
    import tensorflow as tf 
    from tensorflow import keras
    return keras.models.load_model(modelname)

def load_model_yolo(modelname):
    print('[INFO] Loading model YOLO ...{}'.format(modelname))
    configs = os.path.join(modelname, 'yolov3-tiny-staff.cfg')
    weights = os.path.join(modelname, 'yolov3-tiny-staff_best.weights')
    yolonames = os.path.join(modelname, 'yolonames.names')
    net = cv.dnn.readNetFromDarknet(configs, weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    with open(yolonames, 'r') as stream:
        classes = [line.strip() for line in stream.readlines()]
    stream.close()
    print('[INFO] Loaded model YOLO {}'.format(modelname))
    return net, output_layers, classes

def draw_box_text(image, pt1, pt2, string=None):
    # cv.rectangle(image, pt1, pt2, (255,128,125),-1)
    w, h = pt2[0]-pt1[0], pt2[1]-pt1[1]
    cv.putText(image, string,
                org=(pt1[0]+ w*5//8, pt1[1] + h*2//3),
                fontFace= cv.FONT_HERSHEY_SIMPLEX,
                fontScale= h*0.025,
                color=(255,255,255),
                thickness=2,
                lineType=cv.LINE_AA)
    cv.putText(image, string,
                org=(pt1[0]+ w*5//8, pt1[1] + h*2//3),
                fontFace= cv.FONT_HERSHEY_SIMPLEX,
                fontScale= h*0.025,
                color=(0,0,0),
                thickness=1,
                lineType=cv.LINE_AA)
    return image

def drawRect(image, box, color, text = None):
    img = cv.rectangle(img = image,
                        pt1= (box[0], box[1]),
                        pt2= (box[0] + box [2], box[1] + box[3]),
                        color= color,
                        thickness= 2,
                        lineType= cv.LINE_AA)
    if not text == None:
        img = cv.rectangle(img = image,
                            pt1= (box[0], box[1]-12),
                            pt2= (box[0] + box [2], box[1]),
                            color= color,
                            thickness= -1,
                            lineType= cv.LINE_AA)
        img = cv.putText(img= img, text= text,
                        org= (box[0], box[1]-10),
                        fontFace= cv.FONT_HERSHEY_SIMPLEX,
                        fontScale= 0.5,
                        color= (0,0,0),
                        thickness= 1,
                        lineType= cv.LINE_AA)
    return  img

class detect_people():
    COLOR = [(0,0,0), (26,216,93)]
    def __init__(self, modelFolder):
        #model cung ten voi cam id 
        self.name_camera = modelFolder
        path_txt = os.path.join(modelFolder, 'area.txt')
        self.net, self.output, self.classes = load_model_yolo(modelFolder)
        self.tracker = Sort(max_age=80, min_hits=10)
        self.point = irbox.load_area_point(path_txt)
        self.ID = []

    def processing(self, image):
        (height, width) = image.shape[:2]
        points = irbox.fit_size(self.point, width_image=width,
                                height_image=height)
        BG = image.copy()
        BG = irbox.draw_box_area(BG, points)
        blob = cv.dnn.blobFromImage(image=image,
                                    scalefactor=0.00392,
                                    size=(608,608),
                                    mean=(0,0,0),
                                    swapRB=True,
                                    crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output)
        confidences =[]
        boxes =[]
        class_ids = []
        tracking =[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)


                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # box = detection[:4] * np.array([width,height,width,height])
                    # (centerX, centerY, w,h) = box.astype('int')
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.01, 0.5)
        for i in range(len(boxes)):
            if i in indexes:
                (x1,y1,x2,y2) = (boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],boxes[i][1] + boxes[i][3])
                BG = drawRect(BG, boxes[i][:5], self.COLOR[class_ids[i]])
                if class_ids[i] == 1:
                    tracking.append([x1,y1,x2,y2,confidences[i]])
        tracking = np.array(tracking, dtype=np.float32)
        trackers = self.tracker.update(tracking)

        for d in trackers:
            xc, yc = int(d[0] + (d[2]-d[0])/2), int(d[1] + (d[3]-d[1])/2)
            # print(xc,yc, width, height)
            cv.circle(BG, (xc,yc), 5, (0,0,255), -1)
            cv.putText( BG, str(d[4]), (xc,yc),
                        fontFace= cv.FONT_HERSHEY_SIMPLEX,
                        fontScale= 0.5,
                        color= (0,0,0),
                        thickness= 1,
                        lineType= cv.LINE_AA) 
            try:
                self.ID.index(int(d[4]))
            except ValueError:
                if irbox.checkIn((xc,yc), points):
                    self.ID.append(int(d[4]))
            
        text = 'number of customers: {}'.format(len(self.ID))
        BG = draw_box_text(BG,(0,int(height*17/18)), (width, int(height)), string=text)
        return BG 

class detect_peoplev1():
    COLOR = [(0,0,0), (26,216,93)]
    def __init__(self, modelFolder):
        #model cung ten voi cam id 
        self.name_camera = modelFolder
        path_txt = os.path.join(modelFolder, 'area.txt')
        self.net, self.output, self.classes = load_model_yolo(modelFolder)
        self.tracker = Sort(max_age=80, min_hits=10)
        self.point = irbox.load_area_point(path_txt)
        self.ID = []
        
    
    def processing(self, image):
        (height, width) = image.shape[:2]
        points = irbox.fit_size(self.point, width_image=width,
                                height_image=height)
        BG = image.copy()
        BG = irbox.draw_box_area(BG, points)
        return BG
