# import cv2
import cv2

# import time
import time 

#convert BGR TO RGB
def convertToRGB(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#load lbp classifier
lbp_face_cascade = cv2.CascadeClassifier('/home/yashikanand/opencv-3.1.0/data/lbpcascades/lbpcascade_frontalface.xml')

#load test image
test1 = cv2.imread('/home/yashikanand/1.jpg')

# making copy of img
def detect_faces(f_cascade,colored_img,scaleFactor = 1.1):
	img_copy = colored_img.copy()

#convert to gray scale
	gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)

# detection 
	faces = f_cascade.detectMultiScale(gray,scaleFactor = scaleFactor, minNeighbors = 5);
# print number of faces detected
	print "Faces detected:", len(faces)

# draw rectangles across the faces
	for (x,y,w,h) in faces:
		cv2.rectangle(colored_img, (x,y), (x+w,y+h) , (0,0,255) , 10)

	return img_copy
t1 = time.time()

#calling function
faces_detected_img = detect_faces(lbp_face_cascade , test1)
t2 = time.time()

print t2-t1
# convert img to RGB
convertToRGB(test1)

# print ing 
small1 = cv2.resize(test1, (0,0), fx=0.15, fy=0.15) 
cv2.imshow('Test Image', small1)
cv2.waitKey(0)
cv2.destroyAllWindows()
