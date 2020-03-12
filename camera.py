import cv2
import numpy as np 
import os
import imutils

class Camera_thread():
    def __init__(self, queue_ip, queue_name_camera):
        self.queue_ip = queue_ip
        self.queue_name_camera = queue_name_camera

    def get_frame_camera_rtsp(self, source, queue, bool_cam):
        # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        while True:
            camera = cv2.VideoCapture(source)
            print(source)
            count = 1
            while True:
                try:
                    ret, frame = camera.read()
                    
                    if ret is False:
                        break
                    if count % 3 == 0:
                        count+=1
                        continue
                    else:
                        count = 1
                    if ret and queue.empty() is True:
                        frame = imutils.resize(frame, width= 1280)
                        queue.put(frame)
                        bool_cam = True
                except:
                    continue
    
    def get_frame_camera_rtspv1(self, source, queue):
        # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        while True:
            camera = cv2.VideoCapture(source)
            print(source)
            count = 1
            while True:
                try:
                    ret, frame = camera.read()
                    
                    if ret is False:
                        break
                    if count % 3 == 0:
                        count+=1
                        continue
                    else:
                        count = 1
                    if ret and queue.empty() is True:
                        frame = imutils.resize(frame, width= 1280)
                        queue.put(frame)
                        bool_cam = True
                except:
                    continue
