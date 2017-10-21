import cv2

from darkflow.net.build import TFNet
from flask import (Flask, flash, jsonify, redirect, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

options = {"model": "cfg/yolo.cfg",
           "load": "bin/yolo.weights",
           "threshold": 0.3}
tfnet = TFNet(options)

cap = cv2.VideoCapture("IMG_1745.MOV")
#cap.set(cv2.CAP_PROP_POS_MSEC,10000)
#cap.set(cv2.CAP_PROP_POS_FRAMES,300)
#cap.set(cv2.CAP_PROP_FRAME_COUNT,10)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
if(cap.isOpened()):
    print("Video opened")
    c=0
    while(cv2.waitKey(1) != ord('q')):
    #while(c<10):
        ret, frame = cap.read()

        if ret == False:
            print("Video ends")
            break
        frame = cv2.resize(frame,(480,720),interpolation=cv2.INTER_CUBIC)
        predictions = tfnet.return_predict(frame)
        #print(predictions)
        for things in predictions:

        	frame = cv2.rectangle(frame,(things['topleft']['x'],things['topleft']['y']),(things['bottomright']['x'],things['bottomright']['y']),(179, 172, 255),3)
        	
        	cv2.putText(frame,things['label'],(things['topleft']['x'],things['topleft']['y']), font, 1,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow("Video",frame)
        #c=c+1

else:
    print("Opening video failed")

cap.release()
cv2.destroyAllWindows()