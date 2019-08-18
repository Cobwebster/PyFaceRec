import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Mouth.xml')

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter.fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

count = {
    "faces": 0,
    "eyes": 0
}

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile = smile_cascade.detectMultiScale(gray, 7, 7)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        for (sx, sy, sw, sh) in smile:
            cv2.putText(img, 'mouth', (sx-sw, sy-sh), font, 1, (0,255,255), 2, cv2.LINE_AA)
            #cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (120, 172, 255), 2)

        count['eyes'] = len(eyes)
        count['faces'] = len(faces)
    details = "Faces {}  Eyes {}".format(count['faces'], count['eyes'])
    cv2.putText(img, details, (50, 50), font, 1, (0, 175, 255), 2, cv2.LINE_AA)

    cv2.imshow('img', img)
    out.write(img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

out.release()
cv2.destroyAllWindows()
cap.release()
