from flask import Flask,render_template,Response
from backend.normal_camera import normal_camera
from backend.filter_camera import filter_camera

app=Flask(__name__)
camera=cv2.VideoCapture(0)
            

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(normal_camera(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5050,debug=True)