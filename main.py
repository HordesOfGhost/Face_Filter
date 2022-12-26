from flask import Flask,render_template,Response,request
from backend.record import Capture,generate_video
from backend.process import preprocess
import numpy as np
import cv2,io,os,atexit
from PIL import Image
from flask_wtf import FlaskForm
from wtforms import MultipleFileField
from flask_uploads import configure_uploads, IMAGES, UploadSet

app=Flask(__name__)

app.config['UPLOADED_IMAGES_DEST'] = 'images'
app.config['SECRET_KEY']='adadad1212'
app.config['TEMPLATES_AUTO_RELOAD'] = True

images = UploadSet('images', IMAGES)
configure_uploads(app, images)
 



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
            cv2.imwrite('images/glass.jpg',glass)
        except:
            pass
        try:
            cv2.imwrite('images/moustache.jpg',moustache)
        except:
            pass
    return render_template('index.html')

@app.route('/video')
def video():
    try:
        glass=cv2.imread('images/glass.jpg')
        moustache=cv2.imread('images/moustache.jpg')
    except:
        glass=None
        moustache=None
    return Response(generate_video(Capture(),glass=glass,moustache=moustache),mimetype='multipart/x-mixed-replace;boundary=frame')

#On exit delete all filter
def OnExitApp():
    try:
        os.remove('images/glass.jpg')
    except:
        pass
    try:
        os.remove('images/moustache.jpg')
    except:
        pass

atexit.register(OnExitApp)

if __name__=="__main__":
    
    app.run(host='0.0.0.0',port=5050,debug=True)