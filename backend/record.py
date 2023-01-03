import cv2
from datetime import datetime
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
    
    def filter(self,glass=None,moustache=None,save=None):
        #for computation handling
        ret,frame= self.video.read()
        frame=cv2.resize(frame,(640,480))
        frame=cv2.flip(frame,180)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=self.face_detecter(gray,0)
        
        for face in faces:
            #shape
            shape=self.face_landmark(gray,face)
            shape=face_utils.shape_to_np(shape)
            
            #geting cordinates
            nose = shape[self.nose_start:self.nose_end]
            righteyebrow = shape[self.righteyebrow_start:self.righteyebrow_end]
            jaw=shape[self.jaw_start:self.jaw_end]
            
            #for glasses
            glass_left=jaw[16][0]+35
            glass_right=jaw[2][0]-35
            glass_up = righteyebrow[2][1]-25
            glass_down = nose[4][1]+25

            glass_dist_x = glass_left-glass_right
            glass_dist_y = glass_down-glass_up
            
            #glass filter
            if not glass is None:
                if glass_left<=640 and glass_left>=0 and glass_right>=0 and glass_right<640 and glass_up<=480 and glass_down<=480 and glass_up>=0 and glass_down>=0:
                    glass=cv2.resize(glass,(glass_dist_x,glass_dist_y))
                    roi=frame[glass_up:glass_down,glass_right:glass_left]
                    img2gray=cv2.cvtColor(glass,cv2.COLOR_BGR2GRAY)
                    if glass[0][0][0] > 200:
                        _,mask=cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY_INV)
                    else:
                        _,mask=cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY)
                    mask_inv=cv2.bitwise_not(mask)
                    frame_bg=cv2.bitwise_and(roi,roi,mask=mask_inv)
                    glass_fg=cv2.bitwise_and(glass,glass,mask=mask)
                    roi=cv2.add(frame_bg,glass_fg)
                    frame[glass_up:glass_down,glass_right:glass_left]=roi
            nose=shape[self.nose_start:self.nose_end]
            mouth=shape[self.mouth_start:self.mouth_end]
            
            #area to put moustache    
            moustache_right=mouth[0][0]-25
            moustache_left=mouth[6][0]+25
            moustache_up=nose[6][1]-25
            moustache_down=mouth[4][1]+25
            moustache_dist_x=moustache_left-moustache_right
            moustache_dist_y=moustache_down-moustache_up
            
            #moustache filter
            if not moustache is None:    
                if moustache_left<=640 and moustache_left>=0 and moustache_right>=0 and moustache_right<640 and moustache_up<=480 and moustache_down<=480 and moustache_up>=0 and moustache_down>=0:
                    moustache=cv2.resize(moustache,(moustache_dist_x,moustache_dist_y))
                    roi_m=frame[moustache_up:moustache_down,moustache_right:moustache_left]
                    img2gray=cv2.cvtColor(moustache,cv2.COLOR_BGR2GRAY)
                    if moustache[0][0][0] > 200:
                        _,mask=cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY_INV)
                    else:
                        _,mask=cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY)
                    mask_inv=cv2.bitwise_not(mask)
                    frame_bg_m=cv2.bitwise_and(roi_m,roi_m,mask=mask_inv)
                    moustache_fg=cv2.bitwise_and(moustache,moustache,mask=mask)
                    roi_m=cv2.add(frame_bg_m,moustache_fg)
                    roi_m=cv2.add(frame_bg_m,moustache_fg)
                    frame[moustache_up:moustache_down,moustache_right:moustache_left]=roi_m
            
        if save is not None:
            now_time=datetime.now()
            current_time = now_time.strftime("%d_%m_%Y_%H_%M_%S")
            cv2.imwrite(f'saved_snaps/{current_time}.jpg',frame)
            save=None
            
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
    
        

def generate_video(camera,glass=None,moustache=None,save=None):
    while True:
        frame=camera.filter(glass=glass,moustache=moustache,save=save)
        
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

def save_image():
    pass