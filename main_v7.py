import sys
import urllib
import threading
from PIL import Image
import numpy as np
import cv2
import time
import datetime
import requests
from flask import Flask, render_template, Response, send_file, request, json, make_response, jsonify,redirect, url_for
from detectPeoplev5 import detect_people
from multiprocessing import Process, Manager, cpu_count, Queue
import multiprocessing
import logging
from camera import Camera_thread
import serverApiConst as api
from detectPeoplev5 import detect_peoplev1
import detectPeoplev5
import IrregularBox as irbox
from postdatabase import postdata

from flask_cors import CORS, cross_origin

'''pid in ubuntu'''
pid_in = None
pid_out = None
pid_mainprocess = None

''' Running process '''
process_in = None
process_out = None
process_top = None

queue_pid = Queue()
connection_camera_in = Queue(1)
connection_camera_out = Queue(1)
""" address camera """
cam_1 = api.cam_1
cam_2 = api.cam_2
# cam_top = api.CAMERA_TOP
""" model camera """
model_555 = detect_peoplev1('cam554')
model_554 = detect_peoplev1('cam555')
Logging = None


''' Output camera '''
output_cam_CAM1 = Queue(1)
output_cam_CAM2 = Queue(1)

''' Input camera '''
input_cam_CAM1 = Queue(1)
input_cam_CAM2 = Queue(1)


''' information camera '''
ip_camera = Queue(1)
name_camera = Queue(1)
time_now_cam1 = api.timeline
time_now_cam2 = api.timeline

bytes = bytes()
bytes_out = bytes

app = Flask(__name__)
CORS(app)
#initial global variable

FRAME_CAM2 = None
FRAME_CAM1 = None


def read_frame_global():
    """ output processed image """
    global FRAME_CAM1

    global FRAME_CAM2

    while True:
        if not output_cam_CAM2.empty():
            FRAME_CAM1 = output_cam_CAM2.get()
        if not output_cam_CAM1.empty():
            FRAME_CAM2 = output_cam_CAM1.get()

def response_frame_CAM1():
    """ output frame of camera top """
    global FRAME_CAM1
    while True:
        if FRAME_CAM1 is not None:
            _, jpg = cv2.imencode('.jpg', FRAME_CAM1)
            frame = jpg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def response_frame_CAM2():
    """ output frame of camera top """
    global FRAME_CAM2
    while True:
        if FRAME_CAM2 is not None:
            _, jpg = cv2.imencode('.jpg', FRAME_CAM2)
            frame = jpg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run(queue_input, queue_output, name_cam, model_cam, time_now):
    global Logging
    Logging.info("Get frames from id camera {}".format(name_cam))
    
    while True:
        if not queue_input.empty() :
            now = datetime.datetime.now()
            # print(name_cam, 'IN: ',queue_input.qsize())
            frame = queue_input.get()
            height, width = frame.shape[:2]
            point = model_cam.point
            points = irbox.fit_size(point, width_image=width,
                                height_image=height)
            BG = irbox.draw_box_area(frame, points)
            blob = cv2.dnn.blobFromImage(image=frame,
                                    scalefactor=0.00392,
                                    size=(608,608),
                                    mean=(0,0,0),
                                    swapRB=True,
                                    crop=False)
            model_cam.net.setInput(blob)
            outs = model_cam.net.forward(model_cam.output)
            confidences =[]
            boxes =[]
            class_ids = []
            tracking = []
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

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.01, 0.5)
            for i in range(len(boxes)):
                if i in indexes:
                    (x1,y1,x2,y2) = (boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],boxes[i][1] + boxes[i][3])
                    BG = detectPeoplev5.drawRect(BG, boxes[i][:5], model_cam.COLOR[class_ids[i]])
                    if class_ids[i] == 1:
                        tracking.append([x1,y1,x2,y2,confidences[i]])
            tracking = np.array(tracking, dtype=np.float32)
            trackers = model_cam.tracker.update(tracking)

            for d in trackers:
                xc, yc = int(d[0] + (d[2]-d[0])/2), int(d[1] + (d[3]-d[1])/2)
                # print(xc,yc, width, height)
                cv2.circle(BG, (xc,yc), 5, (0,0,255), -1)
                cv2.putText( BG, str(d[4]), (xc,yc),
                            fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 0.5,
                            color= (0,0,0),
                            thickness= 1,
                            lineType= cv2.LINE_AA) 
                try:
                    model_cam.ID.index(int(d[4]))
                except ValueError:
                    if irbox.checkIn((xc,yc), points):
                        model_cam.ID.append(int(d[4]))
                
            text = 'number of customers: {}'.format(len(model_cam.ID))
            BG = detectPeoplev5.draw_box_text(BG,(0,int(height*17/18)), (width, int(height)), string=text)
            queue_output.put(BG)
            if now.hour in time_now:
                if name_cam == 'cam1':
                    name_ID = api.id_store_cam_1
                else:
                    name_ID = api.id_store_cam_2
                data = r'{"storeId":"' + str(name_ID)+ r'","count":' + str(len(model_cam.ID)) + r',"time": "' + now.strftime("%Y-%m-%d") + r' ' + str(now.hour -1)+ r':00"' + r'}'
                Logging.info(data)
                postdata(data)
                time_now.pop(time_now.index(now.hour))
                model_cam.ID = []
            if time_now == []:
                time_now = api.timeline.copy()
            # print(name_cam, 'Out: ',queue_output.qsize())
        else:
            continue
            print('queue empty')
        # Logging.info("Done camera {}".format(name_cam))
#
def run_process_CAM1():
    global process_CAM1
    process_CAM1 = None
    global input_cam_CAM1 , output_cam_CAM1
    global Utils
    global model_554
    Logging.info ("Starting process camera CAM1")

    process_CAM1 = Process(name= 'process_CAM1',target=run, args=(input_cam_CAM1, output_cam_CAM1,'cam1',model_555,time_now_cam1, ))
    process_CAM1.start()
    Logging.info("Started process camera CAM1!")
def run_process_CAM2():
    global process_CAM2
    process_CAM2 = None
    global input_cam_CAM2,output_cam_CAM2
    global model_555

    Logging.info("Starting process camera CAM2")
    process_CAM2 = Process(name= 'process_CAM2',target=run, args=(input_cam_CAM2, output_cam_CAM2,'cam2',model_554,time_now_cam2, ))
    process_CAM2.start()
    Logging.info("Started process camera CAM2!")


"""threading image from camera and put to queu with input_cam_camName """
def thread_input():
    read_camera = Camera_thread(ip_camera, name_camera)
    Logging.info("Starting thread camera CAM1")
    thread_cam_1 = threading.Thread(name='thread_CAM1', target=read_camera.get_frame_camera_rtspv1 ,
                                  args=(cam_1, input_cam_CAM1))
    thread_cam_1.start()
    Logging.info("Started thread camera CAM1!")


    Logging.info("Starting thread camera CAM2")
    thread_cam_2 = threading.Thread(name='thread_CAM2', target = read_camera.get_frame_camera_rtspv1,
                                  args=(cam_2, input_cam_CAM2))
    thread_cam_2.start()
    Logging.info("Started thread camera CAM2!")
def thread_output():
    global data_tablet_in, data_tablet_out
    """ threading output thread cameras cam """
    thread_output = threading.Thread(target=read_frame_global)
    thread_output.start()


@app.before_first_request
def start_cameras():
    """ setup AI models, load server before run routers """
    global Logging
    Logging.info("Login to Dsoft-Backend")

    thread_input()  # running input
    # thread_output()  # running output

    run_process_CAM1()
    run_process_CAM2()
    # run_process_top()

    thread_output()  # running output

def auto_start_app():
    def start_loop():
        global Logging
        # Log
        logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
        Logging = logging.getLogger()
        not_started = True
        Logging.info("Starting flask server...")
        while not_started:
            try:
                r = requests.get(api.AI_TRIGGER_START_FLASK)
                if r.status_code == 200:
                    not_started = False
                    Logging.info("Started flask server!")
            except:
                Logging.error("The Flask server not start yet, keep trying...")
            time.sleep(2)
    thread = threading.Thread(target=start_loop)
    thread.start()
# @app.route('/started', methods=['GET'])
# def is_servered():
#     return Response(status=200)

@app.route('/cam1')
def video_feed():
    return Response(response_frame_CAM1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam2')
def video_out():
    return Response(response_frame_CAM2(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    # t1 = detect_people(cam_2, model_555)
    # t = detect_people(cam_1, model_555)
    auto_start_app()
    app.run(host=api.API_HOST, port=6853, debug=False)

