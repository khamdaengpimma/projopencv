import cv2,os
import numpy
cap = cv2.VideoCapture("video01.mp4")
facedect = cv2.CascadeClassifier("face_model.xml")
def wim(img,id,x,y,w,h):
    im = img[y:y+h,x:x+w]
    cv2.imwrite("kham/image_.1."+str(id)+".jpg",im)
im_id = 0
while True:
    ref,frame = cap.read()
    grat_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedect.detectMultiScale(grat_scale)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        #cv2.putText(frame, "0", (x + 6, y - 6),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
        wim(frame,im_id,x,y,w,h)
        im_id += 1
    #frame = detect(frame,faceCascade)
    cv2.imshow("frame",frame)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()    