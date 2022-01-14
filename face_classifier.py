#recognize faces suing some  classification algo -here,knn

#1 load the trainind data
#   x-values are stored in numpy arrays
#   y-values we need to assign for each person(A,B,c)

#2 read video stream using open cv
#3 extract faces out of it
#4 use knn to find predicitoin of te face (int)
#5 map the predicted id(0,1,2) to name of the user (0->A , 1->B , 2->C) Dictionary mapping
#5 Dsipaly prediction on screen - bounding box and name



import cv2
import numpy as np
import os

#######KNN CODE
#this version takes training,testing data in single. Means X_train & Y_train in one -> 'train'
def distance(v1,v2):
	#Eucledian
	return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
	dist=[]

	for i in range(train.shape[0]):

		#Get the vector and label
		ix=train[i,:-1] #ith row and all columns
		iy=train[i, -1] #ith row ,and it has single column
		#Compute the dist from test point
		d=distance(test,ix)
		dist.append([d, iy])
	#Sort based on distance adn get top k
	dk=sorted(dist,key=lambda x:x[0])[:k]
	#Retrieve only the labels
	labels=np.array(dk)[:,-1]

	#Get frequencies of each label
	output=np.unique(labels,return_counts=True)
	#Find max frequency and corresponding label
	index=np.argmax(output[1])
	return output[0][index]


#############KNN ENDS 

#init camera
cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)

#face detection
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0
#facial data storage
face_data=[]
#labels will form the y labels for knn 
labels=[]
dataset_path='./data/'

class_id=0   #Labels for the given file
names={}    #Mapping bt id-name

#Data Preperation
 #its like dir in  cmd. it lists filenames in the current directory
 #searches for .npy file in give path .here fx wud be the file name

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#Create a mapping btw class_i and name
		names[class_id]=fx[:-4]
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)

		#Create labels for the class
		target=class_id*np.ones((data_item.shape[0],))  #target is a matrix of ones multiplied with cass_id
		class_id+=1
		labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

#We combine the dataset with labels bcoz , our knn takes one 'train' dataset.
#labels column is added after dataset becasue ,knn algo extracts label from last column
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


#Testing

while True:
	ret,frame=cap.read()
	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5)
	
	for face in faces:
		x,y,w,h=face

		#Get the Face ROI
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		#flatten bcoz np array like pixel flat krke deni
		#Predicted label (out)
		out=knn(trainset,face_section.flatten())
  	
  		#Dispaly on the screen he name and rectangle around it
 		#from the dictionary u read the name,lets say out=1, and B->1, so name =B
		pred_name=names[int(out)]
  		#putText takes -> frame,name printed,location(coord),font type,font scale(size),and finally (color,thickness,lineType) we use one command 'cv2.LINE_AA' 
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key=cv2.waitKey(1) &0xFF
	if key==ord('q'):
  		break

cap.release()
cv2.destroyAllWindows()