# import libraries
import time
# starting time
t1 = time.time()

import cv2
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# labels 
subjects = ["", "Yashik","Manish"]

# mean values of RGB
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age list approximations defied
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

# Gender list created
gender_list = ['Male', 'Female']

# Reading complete names dataset
df_names = pd.read_csv('/home/yashikanand/project/Gender_classification/names_dataset.csv')

# Replacing Gender in integer format
df_names.sex.replace({'F':0, 'M':1},inplace=True)

# extracting letters from the name
def features(name):
    return {
        'first-letter': name[0], 
        'first2-letters': name[0:2],
        'first3-letters': name[0:3],
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

# vectorizing the extracted letters
features = np.vectorize(features)

# creating dataset
df_X = features(df_names['name'])
df_y = df_names['sex']

# creating train and test dataset
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

dv = DictVectorizer()

dv.fit_transform(dfX_train)

# training decisiontreeclassifier 
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)

# gender predictor
def genderpredictor(a):
	# creating array of name
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()

    # checking the gender
    if dclf.predict(vector) == 0:
        X = "Female"
    else:
        X = "Male"
    return X

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
		if not dir_name.startswith("s"):
			continue;
	# extract label number of subject from dir_name
	# format of dir name = slabel

		label = int(dir_name.replace("s" , ""))

	#build path of directory containing images for current subject
		subject_dir_path = data_folder_path + "/" + dir_name

	#get the img names that are inside the  given subject directory
		subject_images_names = os.listdir(subject_dir_path)
	
	# go through image name, read image,
		# detect face and add face to list of faces
		for image_name in subject_images_names:

		#ignore system files 
			if image_name.startswith("."):
				continue;
	
	#build image path
			image_path = subject_dir_path + "/" + image_name

	#read image
			image = cv2.imread(image_path)

	#detect face
			face, rect = detect_face(image)

	# ignore faces that are not detected
			if face is not None:
				faces.append(face)
				labels.append(label)

	return faces , labels

# preparing the data from the source
print ("Preparing data...")
faces , labels = prepare_training_data("/home/yashikanand/project/training-data")
print ("Data prepared")

#print total faces and labels
print "Total faces: ", len(faces)
print "Total labels: ", len(labels)


#create face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# train face recognizer
face_recognizer.train(faces,np.array(labels))

#function to draw rectangle
def draw_rectangle(img, rect):
	(x,y,w,h) = rect
	cv2.rectangle(img, (x,y), (x+w , y+h), (0,255,0),10)
	return (x,y,w,h)

#function to draw text
def draw_text(img,text,x,y):

	cv2.putText(img , text, (x,y), cv2.FONT_HERSHEY_PLAIN, 5,(0,255,0), 10)

#function to recognize the face in img passed and draw a rectangle with name
def predict(test_img):

	# make a copy
	img = test_img.copy()

	# detect face from the img
	face , rect = detect_face(img)

	# predict the image using recognizer
	label,confidence = face_recognizer.predict(face)
	#get name
	label_text = subjects[label]
	
	# predicting the gender
	gender = genderpredictor(label_text)

	# predicting the age
	age = age_predictor(img,age_net,rect)

	label = label_text +"," + gender +"," + age
	
	rect1 = draw_rectangle(img,rect)
	draw_rectangle(img,rect)
	draw_text(img, label, rect[0],rect[1]-5)
	#draw_text(img, gender, rect[0],rect[1]-5)

	return img

# initalizing caffe model for age classifier
def initialize_caffe_model():
    age_net = cv2.dnn.readNetFromCaffe(
                        "/home/yashikanand/project/Age_classification/deploy_age.prototxt", 
                        "/home/yashikanand/project/Age_classification/age_net.caffemodel")
    return (age_net)

# for predicting the age
def age_predictor(image,age_net,rect): 
    (x,y,h,w) = rect
    face_img = image[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
        
    return age          
    
print "Predicting images..."

#initialing the caffe model
age_net = initialize_caffe_model()

#load test images
test_img1 = cv2.imread("/home/yashikanand/project/test-data/test1.jpg")

# prediction
predicted_img1 = predict(test_img1)

print "Prediction complete"

# accuracy of the gender predictor
print ('Accuracy of train results = ')
print(dclf.score(dv.transform(dfX_train), dfy_train)) 
print ('Accuracy of test results = ')
print(dclf.score(dv.transform(dfX_test), dfy_test))

#display images
small1 = cv2.resize(predicted_img1, (0,0), fx=0.5, fy=0.5) 
cv2.imshow(subjects[1], small1)
t2 = time.time()
print 'Time taken for recognition = ' 
print t2-t1
cv2.waitKey(0)
cv2.destroyAllWindows()
