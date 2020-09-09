# import cv2
import cv2

# import time
import time 

#convert BGR TO RGB
def convertToRGB(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# load classifier
haar_face_cascade = cv2.CascadeClassifier('/home/yashikanand/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')


# making copy of img
def detect_faces(f_cascade,colored_img,scaleFactor = 2.5):
	img_copy = colored_img.copy()

#convert to gray scale
	gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)

# detection 
	faces = f_cascade.detectMultiScale(gray,scaleFactor = scaleFactor, minNeighbors = 5);
# print number of faces detected
	print "Faces detected:", len(faces)
	
# draw rectangles across the faces
	for (x,y,w,h) in faces:
		cv2.rectangle(colored_img, (x,y), (x+w,y+h) , (0,255,0) , 10)

	return img_copy

#load another image
test2 = cv2.imread('/home/yashikanand/2.jpg')

t1 = time.time()

#calling function
face_detected_img = detect_faces(haar_face_cascade,test2)

t2 = time.time()

print t2-t1
# convert img to RGB
convertToRGB(test2)

# print ing 
small2 = cv2.resize(test2, (0,0), fx=0.15, fy=0.15) 
cv2.imshow('Test Image', small2)
cv2.waitKey(0)
cv2.destroyAllWindows()
