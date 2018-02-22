from PIL import Image
import numpy as np
import cv2
from datetime import datetime

def diffImg(t0, t1, t2):
	  d1 = cv2.absdiff(t2, t1)
	  d2 = cv2.absdiff(t1, t0)
	  return cv2.bitwise_and(d1, d2)

threshold = 152500
cam = cv2.VideoCapture(0)
winName = "Movement Indicator"
cv2.namedWindow(winName)

t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
timeCheck = datetime.now().strftime('%Ss')
d = 1
p = 1
while True:
	  cv2.imshow( winName,cam.read()[1])
	  if cv2.countNonZero(diffImg(t_minus,t,t_plus)) > threshold and timeCheck != datetime.now().strftime('%Ss'):
	 	 dimg= cam.read()[1]
		 filename = "pic_%d.jpg"%p
		 cv2.imwrite(filename,dimg)
	  	 
		 faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
		 img = cv2.imread("pic_%d.jpg"%p)
		 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		 p+=1

		 faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
		 print "Found {0} faces!".format(len(faces))
		 
		 # Draw a rectangle around the faces
		 for (x, y, w, h) in faces:
		    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 25), 2)
		    cropped = img[y:y+h,x:x+w]
		    filename = "cropped_%d.jpg"%d
		    cv2.imwrite(filename,cropped)
		    d+=1
		  
		 #cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f')+'.jpg',dimg)
	  timeCheck = datetime.now().strftime('%ss')
	  # Read next image
	  t_minus = t
	  t = t_plus
	  t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
	  key = cv2.waitKey(10)
	  if key == 27:
	    cv2.destroyWindow(winName)
	    break


