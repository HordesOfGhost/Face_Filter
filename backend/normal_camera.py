import cv2
import time
camera=cv2.VideoCapture(0)

def normal_camera(): 
    pTime=0
    while True:
        sucess,frame=camera.read()
        frame=cv2.flip(frame,180)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(frame,f'FPS: {int(fps)}',(30,70),cv2.FONT_HERSHEY_PLAIN,
                    3,(0,255,0),3)
        if not sucess:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
