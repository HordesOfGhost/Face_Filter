import cv2
import dlib
from imutils import face_utils
import numpy as np
import os
def filter_camera(glass=None,moustache=None):
    #For Universal Path
    path=os.path.dirname(os.path.realpath(__file__))
    path=path.replace('\\','/')
    
    #face detector and landmarks
    face_detecter=dlib.get_frontal_face_detector()
    face_landmark=dlib.shape_predictor(f'{path}/model/shape_predictor_68_face_landmarks.dat')
    mouth_start,mouth_end=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
    nose_start,nose_end=face_utils.FACIAL_LANDMARKS_IDXS['nose']
    jaw_start,jaw_end=face_utils.FACIAL_LANDMARKS_IDXS['jaw']
    righteyebrow_start,righteyebrow_end = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']

    #for landmarks points
    landmarks_points=[]
    cap=cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()
        
        #setting frame size
        frame=cv2.resize(frame,(1280,720))
        frame=cv2.flip(frame,180)
        #for computation handling
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_detecter(gray,0)
        
        #For corrdinates
        for face in faces:
            
            shape=face_landmark(gray,face)
            shape=face_utils.shape_to_np(shape)
            
            #for nose and mouth
            nose=shape[nose_start:nose_end]
            mouth=shape[mouth_start:mouth_end]
            
            #area to put moustache    
            right=mouth[0][0]-50
            left=mouth[6][0]+50
            up=nose[6][1]-50
            down=mouth[4][1]+50
            dist_x1_m=left-right
            dist_y1_m=down-up
            
            #Prevents Crashing    
            if left<=1280 and left>=0 and right>=0 and right<1280 and up<=720 and down<=720 and up>=0 and down>=0:
                #reading and resizing
                moustache=cv2.imread(f'{path}/images/moustache.jpg')
                moustache=cv2.resize(moustache,(dist_x1_m,dist_y1_m))
                    
                #setting Moustache frame   
                roi_m=frame[up:down,right:left]
                    
                #converting to gray scale
                img2gray=cv2.cvtColor(moustache,cv2.COLOR_BGR2GRAY)
                    
                #masking so that the background is all white # inverse because moustache is all black
                _,mask=cv2.threshold(img2gray,50,255,cv2.THRESH_BINARY_INV)
                    
                #inversing mask so that subject is all white
                mask_inv=cv2.bitwise_not(mask)
                    
                #this ensures only moustache  is read
                frame_bg_m=cv2.bitwise_and(roi_m,roi_m,mask=mask_inv)
                    
                #this now ensures background to go
                moustache_fg=cv2.bitwise_and(moustache,moustache,mask=mask)
                
                #adds bg and fg    
                roi_m=cv2.add(frame_bg_m,moustache_fg)
                    
                #adds to frame    
                roi_m=cv2.add(frame_bg_m,moustache_fg)
                    
                frame[up:down,right:left]=roi_m
                
            #for eyes
            
            nose = shape[nose_start:nose_end]
            righteyebrow = shape[righteyebrow_start:righteyebrow_end]
            jaw=shape[jaw_start:jaw_end]

            #for glasses
            left1=jaw[16][0]+70
            right1=jaw[2][0]-70
            up1 = righteyebrow[2][1]-50
            down1 = nose[4][1]+50

            dist_x1 = left1-right1
            dist_y1 = down1-up1
                
            #Prevents Crashing
            if left1<=1280 and left1>=0 and right1>=0 and right1<1280 and up1<=720 and down1<=720 and up1>=0 and down1>=0:
                #reading and resizing
                glass=cv2.imread(f'{path}/images/glass.jpg')
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
        
        #showing frames or video    
        if ret:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')