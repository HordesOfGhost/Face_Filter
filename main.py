from flask import Flask,render_template,Response,request,redirect
from backend.record import Capture,generate_video,save_image
from backend.pre_process import preprocess
import numpy as np
import cv2,io,os,atexit
from PIL import Image

app=Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=="POST":
        #Read Glass Filter
        for upload in request.files.getlist("glass"):
            glass=np.array(bytearray(upload.read()),dtype=np.uint8)
        #Read Moustache Filter
        for upload in request.files.getlist("moustache"):
            moustache=np.array(bytearray(upload.read()),dtype=np.uint8)
        
        #Preprocess
        glass=preprocess(glass)
        moustache=preprocess(moustache)
        
        #Save
        try:
            cv2.imwrite('backend/temp_images/glass.jpg',glass)
        except:
            pass
        try:
            cv2.imwrite('backend/temp_images/moustache.jpg',moustache)
        except:
            pass
    return render_template('index.html')

@app.route('/video')
def video():
    try:
        glass=cv2.imread('backend/temp_images/glass.jpg')
        moustache=cv2.imread('backend/temp_images/moustache.jpg')
    except:
        glass=None
        moustache=None
    return Response(generate_video(Capture(),glass=glass,moustache=moustache,save=None),mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/remove_glass/')
def Remove_Glass():
    try:
        os.remove('backend/temp_images/glass.jpg')
    except:
        pass
    return redirect('/')

@app.route('/remove_moustache/')
def Remove_Moustache():
    try:
        os.remove('backend/temp_images/moustache.jpg')
    except:
        pass
    return redirect('/')

@app.route('/remove_all/')
def Remove_all():
    try:
        os.remove('backend/temp_images/glass.jpg')
    except:
        pass
    try:
        os.remove('backend/temp_images/moustache.jpg')
    except:
        pass
    return redirect('/')

@app.route('/snapshot/')
def Snapshot():
    try:
        glass=cv2.imread('backend/temp_images/glass.jpg')
        moustache=cv2.imread('backend/temp_images/moustache.jpg')
    except:
        glass=None
        moustache=None
    #To Capture
    Capture().filter(glass=glass,moustache=moustache,save=1)
   
    return redirect('/')

#On exit delete all filter
def OnExitApp():
    try:
        os.remove('backend/temp_images/glass.jpg')
    except:
        pass
    try:
        os.remove('backend/temp_images/moustache.jpg')
    except:
        pass

atexit.register(OnExitApp)


if __name__=="__main__":
    
    app.run(host='0.0.0.0',port=5050,debug=True)