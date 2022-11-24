import cv2
import numpy
cap = cv2.VideoCapture(0)
facedect = cv2.CascadeClassifier("face_model.xml")
def wim(img,id,x,y,w,h):
    im = img[y:y+h,x:x+w]
    cv2.imwrite("kham/kham.1."+str(id)+".jpg",im)
imd = 0
while True:
    ref,frame = cap.read()
    grat_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedect.detectMultiScale(grat_scale)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        #cv2.putText(frame, "0", (x + 6, y - 6),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
        k = cv2.waitKey(1)
        if k%256 == 27:
        # ESC pressed
            print("Escape hit, closing…")
            break
        elif k%256 == 32:
        # SPACE pressed
            wim(frame,imd,x,y,w,h)
            imd +=1
    cv2.imshow("frame",frame)
cap.release()
cv2.destroyAllWindows()    