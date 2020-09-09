# import cv2
import cv2

# import time
import time 

#convert BGR TO RGB
def convertToRGB(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#load img
test1 = cv2.imread('/home/yashikanand/1.jpg')

#convert to gray scale
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)


# display image
small = cv2.resize(gray_img, (0,0), fx=0.25, fy=0.25) 

# load classifier
haar_face_cascade = cv2.CascadeClassifier('/home/yashikanand/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

t1 = time.time()
# detect faces
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5);
t2 = time.time()


print t2-t1
#print the number of faces found
print 'Faces found:', len(faces)

# draw rectangles around face
for (x,y,w,h) in faces:
	cv2.rectangle(test1, (x,y), (x+w,y+h) , (0,255,0) , 10)

# convert to RGB
convertToRGB(test1)

# print ing 
small1 = cv2.resize(test1, (0,0), fx=0.25, fy=0.25) 
cv2.imshow('Test Image', small1)
cv2.waitKey(0)
cv2.destroyAllWindows()

