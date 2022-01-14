#1 capture image from video cam
#2 detect faces and show bounding boxes using haarcascade
#3 flatten the largest image (gray scale) into numpy array
#4 repeat the above for mutiple people to generate training datas

import cv2
import numpy as np
#init camera

#below is depricated one ->shows error
#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)

#face detection
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0
#facial data storage
face_data=[]
dataset_path='./data/'
#Filename wil be persons name
file_name=input('Enter the name of the person : ')

while True:
	ret,frame=cap.read()
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
	#print(faces)
	#this wud print the coord of faces in realtime

	#we need largest face, so we sort it.And for largest we'd need to multiply width*height, so [area= f[2]*f[3] ] because  faces returns x,y,w,h
	
	faces=sorted(faces,key=lambda f:f[2]*f[3])


	#for multiple faces 
	#pick the last face because it is the largest area acc to f[2]*f[3]
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of interest
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		#offset=10pixels -> we slice out the required part with a padding of 10 pixels
		#and resize it to 100x100 pixels
		face_section=cv2.resize(face_section,(100,100))


		#we'd save only the tenth 10th frame. We'd save face section into face_data 
		skip+=1
		#har frame pe increment hota and har 10th frame capture hota
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
			#we also show how many frmaes captured yet

	#display the frame 
	cv2.imshow('Video Frame',frame)
	cv2.imshow('Face Section',face_section)

 	
	#wait for user input ->q , thenyou'll stop the loop
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break

#Convert our face lsit array into numpy array
face_data=np.asarray(face_data)
#no of rows=no of faces and columns figured out automatically
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
#itd be saved in /data/bhuan.npy
#npy is file format for numpy
print('Data successfully saved @'+dataset_path+file_name+'.npy' )


cap.release()
cv2.destroyAllWindows()
