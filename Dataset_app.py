# import libraries
import cv2
import os
import numpy as np
import time

# function to detect face
def detect_face(img):

	# convert img to gray
	gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

	# load cv2 face detector
	face_cascade = cv2.CascadeClassifier('/home/yashikanand/opencv-3.1.0/data/lbpcascades/lbpcascade_frontalface.xml')

	# detecting faces
	faces = face_cascade.detectMultiScale(gray , scaleFactor = 1.2, minNeighbors = 5)

	#if no face detected
	if (len(faces) == 0):
		return None,None

	# if only one face
	# extract face
	(x,y,w,h) = faces[0]

	# return face part
	return gray[y:y+w, x:x+h] , faces[0]
subjects = [""]
# function will read faces from training img
# and detect face and retun two lists with same size, 
# one with face data and other with labels
def prepare_training_data(data_folder_path):
	
	# get the directories in data folder
	dirs = os.listdir(data_folder_path)

	# list to hold the data
	faces = []

	labels = []

	
	# let's go through each directory and read images within it
	for dir_name in dirs:
		subjects = dir_name
		label = int("1")

	#build path of directory containing images for current subject
		subject_dir_path = data_folder_path + "/" + dir_name

	#get the img names that are inside the  given subject directory
		subject_images_names = os.listdir(subject_dir_path)
	
	# go through image name, read image,
		# detect face and add face to list of faces
		for image_name in subject_images_names:

			if image_name.startswith("."):
						continue;

			image_name1 = subject_dir_path + "/" + image_name

			image_name1_names = os.listdir(image_name1)

			for image_name2 in image_name1_names:

				if image_name2.startswith("."):
						continue;

				image_name3 = image_name1 + "/" + "images"

				image_name3_names = os.listdir(image_name3)
		
				for image_name4 in image_name3_names:
				#ignore system files 
					if image_name4.startswith("."):
						continue;
	
	#build image path
					image_path = image_name3 + "/" + image_name4


	#read image
					image = cv2.imread(image_path)

	#detect face
					face, rect = detect_face(image)

	# ignore faces that are not detected
					if face is not None:
						faces.append(face)
						labels.append(label)

	return faces , labels

#function to draw rectangle
def draw_rectangle(img, rect):
	(x,y,w,h) = rect
	cv2.rectangle(img, (x,y), (x+w , y+h), (0,255,0),10)

#function to draw text
def draw_text(img,text,x,y):
	cv2.putText(img , text, (x,y), cv2.FONT_HERSHEY_PLAIN, 14,(0,255,0), 2)

#function to reconize the face in img passed and draw a rectangle with name
def predict(test_img):

	# make a copy
	img = test_img.copy()
	# detect face from the img
	face , rect = detect_face(img)

	# predict the image using recognizer
	label,confidence = face_recognizer.predict(face)
	#get name
	label_text = subjects[label]

	draw_rectangle(img,rect)

	draw_text(img, label_text, rect[0],rect[1]-5)

	return img

t1 = time.time()
print ("Preparing data...")
faces , labels = prepare_training_data("/home/yashikanand/Project stuff/IMFDB/IMFDB_final")
print ("Data prepared")

#print total faces and labels
print "Total faces: ", len(faces)
print "Total labels: ", len(labels)


#create face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# train face recognizer
face_recognizer.train(faces,np.array(labels))

print "Predicting images..."

#load test images
test_img1 = cv2.imread("/home/yashikanand/Project stuff/IMFDB/IMFDB_final/AmitabhBachchan/Anand/images/Amitabh_3.jpg")

# prediction
predicted_img1 = predict(test_img1)

print "Prediction complete"

t2 =time.time()
print t2-t1
#display images
#small1 = cv2.resize(predicted_img1, (0,0), fx=0.15, fy=0.15) 
cv2.imshow(subjects[1], predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
