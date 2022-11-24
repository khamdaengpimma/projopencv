import cv2
import numpy as np
cap = cv2.imread('dect.jpg')
facedect = cv2.CascadeClassifier("face_model.xml")
grat_scale = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
faces = facedect.detectMultiScale(grat_scale)
for(x,y,w,h) in faces:
    cv2.rectangle(cap,(x,y),(x+w,y+h),(255,255,0),2)
#frame = detect(frame,faceCascade)
    cv2.imshow("frame",cap)
    #cv2.imwrite("./write/image.jpg",cap)

    
cv2.waitKey(0)
#cap.release()
cv2.destroyAllWindows()    