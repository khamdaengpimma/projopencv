import cv2
import numpy
cap = cv2.VideoCapture(0)
facedect = cv2.CascadeClassifier("hc1_2MB.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("test01.xml")
while True:
    ref,frame = cap.read()
    grat_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedect.detectMultiScale(grat_scale)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        id,_= clf.predict(grat_scale[y:y+h,x:x+w])
        if (_<=100):
            pc = "  {0}%".format(round(100-_))
            pl = 100 - _
        else:
            pc = "  {0}%".format(round(_-100))
            pl = _-100
        #print(str(_))
        if pl>57:
            cv2.putText(frame,'KD'+str(pc), (x + 6, y - 6),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
        #elif pl<50:
            #cv2.putText(frame, "Unknow", (x + 6, y - 6),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Unknow", (x + 6, y - 6),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
        #print(str(pc))
    cv2.imshow("frame",frame)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()    