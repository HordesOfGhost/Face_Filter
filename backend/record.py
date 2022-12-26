import cv2
import time
import dlib
from imutils import face_utils
import numpy as np
import os

camera=cv2.VideoCapture(0)

class Capture():
    def __init__(self):
        self.video=cv2.VideoCapture(0)

        self.path=os.path.dirname(os.path.realpath(__file__))
        self.path=self.path.replace('\\','/')
        
        #face detector and landmarks
        self.face_detecter=dlib.get_frontal_face_detector()
        self.face_landmark=dlib.shape_predictor(f'{self.path}/model/shape_predictor_68_face_landmarks.dat')
        self.mouth_start,self.mouth_end=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        self.nose_start,self.nose_end=face_utils.FACIAL_LANDMARKS_IDXS['nose']
        self.jaw_start,self.jaw_end=face_utils.FACIAL_LANDMARKS_IDXS['jaw']
        self.righteyebrow_start,self.righteyebrow_end = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret,frame= self.video.read()
        frame=cv2.flip(frame,180)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
    
    def filter(self,glass):
        #for computation handling
        ret,frame= self.video.read()
        frame=cv2.resize(frame,(1280,720))
        frame=cv2.flip(frame,180)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=self.face_detecter(gray,0)
        
        for face in faces:
            
            shape=self.face_landmark(gray,face)
            shape=face_utils.shape_to_np(shape)
            
            nose = shape[self.nose_start:self.nose_end]
            righteyebrow = shape[self.righteyebrow_start:self.righteyebrow_end]
            jaw=shape[self.jaw_start:self.jaw_end]
            
            #for glasses
            left1=jaw[16][0]+70
            right1=jaw[2][0]-70
            up1 = righteyebrow[2][1]-50
            down1 = nose[4][1]+50

            dist_x1 = left1-right1
            dist_y1 = down1-up1
            if left1<=1280 and left1>=0 and right1>=0 and right1<1280 and up1<=720 and down1<=720 and up1>=0 and down1>=0:
                #reading and resizing
                #glass=cv2.imread(glass)
                glass=cv2.resize(glass,(dist_x1,dist_y1))
                    
                #setting Glass frame
                roi=frame[up1:down1,right1:left1]
                    
                #converting to gray scale
                img2gray=cv2.cvtColor(glass,cv2.COLOR_BGR2GRAY)
                
                #masking so that the background is all black
                _,mask=cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY)
                #cv2.imshow('mask',mask)
                    
                #inversing mask so that subject is all black
                mask_inv=cv2.bitwise_not(mask)
                    
                #this ensures only colored glass is read
                frame_bg=cv2.bitwise_and(roi,roi,mask=mask_inv)
                    
                #this now ensures background to go
                glass_fg=cv2.bitwise_and(glass,glass,mask=mask)
                    
                #adds bg and fg   
                roi=cv2.add(frame_bg,glass_fg)
                
                #adds to frame   
                frame[up1:down1,right1:left1]=roi
    
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
    
        

def generate_video(camera):
    while True:
        frame=camera.get_frame()
        
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

def filtered_video(camera,glass):
    while True:
        frame=camera.filter(glass=glass)
        
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')