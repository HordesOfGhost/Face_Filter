from flask import Flask,render_template,Response,request
from backend.record import Capture,generate_video,filtered_video
from backend.process import preprocess
import numpy as np
import cv2
from PIL import Image

app=Flask(__name__)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video',methods=['GET','POST'])
def video():
    return Response(generate_video(Capture()),mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/',methods=['POST'])
def filter():
    if request.method == 'POST':
        for upload in request.files.getlist("glass") and request.files.getlist("glass"):
            glass_array=np.array(bytearray(upload.read()),dtype=np.uint8)
        for upload in request.files.getlist("moustache"):
            moustache_array=np.array(bytearray(upload.read()),dtype=np.uint8)
        glass=preprocess(glass_array)
        moustache=preprocess(moustache_array)
        return Response(filtered_video(Capture(),glass=glass,moustache=moustache),mimetype='multipart/x-mixed-replace;boundary=frame')
    return 'not submitted'   


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5050,debug=True)