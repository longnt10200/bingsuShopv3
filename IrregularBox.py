# Chuong trinh tao khu vuc voi hinh dang tu giac bat ki
# va kiem tra 1 diem co nam trong khu vuc khong
import cv2 
import numpy as np 
import datetime
import os
LINK1 = 'rtsp://dsoft:Dsoft@321@113.176.195.116:555/ch1/main/av_stream'
LINK2 = 'rtsp://dsoft:Dsoft@321@113.176.195.116:554/ch1/main/av_stream'
'''Tao khu vuc xac dinh
input: 
    link: duong dan video hoac stream 
    path_save_txt: duong dan va ten tep luu toa do vi tri
Vidu:
test = Setup_box(LINK1)
test.get_point()

toa do tu giac duoc luu o dang ti so
x = x/w
y = y/h

De phu hop voi cac khung hinh khac nhau
''' 
class Setup_box:
    def __init__(self, link, pathTXT):
        self.output_txt = pathTXT
        self.stream = open(self.output_txt, 'w+')
        self.click_count = 0
        self.point = []
        vs = cv2.VideoCapture(link)
        result, frame = vs.read()
        self.image = frame.copy()
        self.backup = frame.copy()
        self.size = frame.shape[:2]
        cv2.namedWindow('Select Space', cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback('Select Space', self.mouse_callback)
        cv2.imshow('Select Space', self.image)
    def get_point(self):
        while True:
            key = cv2.waitKey(5) & 0xFF
            if key == ord('r'):
                self.image = self.backup
                self.point = []
            elif key == ord('q'):
                self.stream.close()
                break
        cv2.destroyWindow('Select Space')

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append((x,y))
            print(self.point)
            # self.click_count +=1
            if len(self.point )==1:
                cv2.circle(self.image, self.point[-1], 10, (0,0,255), -1)
                print('=1')
            elif len(self.point) == 4:                
                points =''
                cv2.line(self.image, self.point[-1], self.point[-2], (0,255,0), 5)
                cv2.line(self.image, self.point[-1], self.point[0], (0,255,0), 5)
                for xr,yr in self.point:
                    xr = float(xr/self.size[1])
                    yr = float(yr/self.size[0])
                    points += '{},{}\n'.format(xr,yr)
                self.stream.write(points)
                print(points)
            else:
                cv2.line(self.image, self.point[-2], self.point[-1], (0,255,0), 5)
            cv2.imshow('Select Space', self.image)
'''
Ham lay toa do cua tu giac
Doc file Txt 
Lay toa do
Chuyen thanh toa do thuc
Input: 
    path: Duong dan file txt
    width_image: chieu rong khung hinh
    height_image: chieu rong khung hinh.
Output: 
    points: list toa do cac diem
'''
def load_area_point(path):
    stream = open(path, 'r')
    points = []
    for line in stream.readlines():
        line = line.strip()
        x, y = line.split(',')
        xb = float(x)
        yb = float(y)
        points.append((xb,yb))
    return points
def fit_size(points, width_image=1280, height_image=720):
    point = []
    for x,y in points:
        xb = int(x * width_image)
        yb = int(y * height_image)
        point.append((xb,yb))
    return point
'''
Ve lai khu vuc tren khung hinh
'''
def draw_box_area(image, points):
    mask = np.zeros_like(image)
    points = np.array(points, dtype='int32')
    if len(image.shape)>2:
        mask_color = (255,0,0)
    else:
        mask_color = 128
    cv2.fillPoly(mask, [points],mask_color)
    image = cv2.addWeighted(image, 0.7, mask, 0.3, 0.6)
    return image
'''
Kiem tra 1 diem nam trong hay ngoai khu vuc quan tam
'''
def cross(o,a,b):
    return (a[0] - o[0]) * (b[1] - o[1]) - \
        (a[1] - o[1]) * (b[0] - o[0])
'''
Input: 
    point: toa do diem can xet
    box: toa do cac diem cua khu vuc
Output:
    inside: boolean, diem do cac nam trong khu vuc khong
    inside == TRUE: Nam trong 
    inside == FALSE: Nam ngoai

vidu:
p = (525,161)
image = cv2.imread('text.png')
(h,w) = image.shape[:2]
points = load_area_point('area.txt', w, h)
out = draw_box_area(image, points)
checkIn(p, points)
cv2.circle(out, p, 10, (0,0,255), -1)
cv2.namedWindow('out', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
def checkIn(point, box):
    inside = True
    for index in range(0, len(box)):
        res = cross(box[index-1], box[index], point)
        # print("cross: ", res)
        if res <0:
            inside = False
    return inside

# test = Setup_box(LINK2, 'cam554/area.txt')
# test.get_point()
# p = (2009,550)
# image = cv2.imread('test2.jpg')
# (h,w) = image.shape[:2]
# points = load_area_point('cam554/area.txt')
# po = fit_size(points, w, h)
# out = draw_box_area(image, po)
# print("INS" if checkIn(p, po) else "OUT", (h,w))
# cv2.circle(out, p, 10, (0,0,255), -1)
# cv2.namedWindow('out', cv2.WINDOW_GUI_EXPANDED)
# cv2.imshow('out', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()