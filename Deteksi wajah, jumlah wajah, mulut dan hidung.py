import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier =cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
	ret, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	font=cv2.FONT_HERSHEY_SIMPLEX
	jumlah=0


	for (x,y,w,h) in faces:
		jumlah=jumlah+1
		cv2.putText(img,"Wajah",(x,y-10),font,0.75,(0,0,255),2,cv2.LINE_AA)
		cv2.putText(img,"Jumlah wajah ada : "+str(jumlah)+ " buah",(10,30),font,1,(0,0,0),2,cv2.LINE_AA)
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_classifier.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


		cv2.imshow('img',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()

