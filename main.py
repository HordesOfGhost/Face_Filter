from flask import Flask,render_template,Response,request
#from flask_wtf import FlaskForm
from backend.record import Capture,generate_video,filtered_video
from backend.filter_camera import filter_camera

app=Flask(__name__)
            

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video',methods=['GET','POST'])
def video():
    return Response(generate_video(Capture()),mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/',methods=['POST'])
def filter():
    if request.method == 'POST':
        txt=str(request.form['glass'])
        return Response(filtered_video(Capture()),mimetype='multipart/x-mixed-replace;boundary=frame')
    return 'not submitted'   


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5050,debug=True)