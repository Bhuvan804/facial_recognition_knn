Facial Recognition Script. </br>
OpenCV is used to take input from webcam and KNN(k=5,Eucledian) is used for building face classifier. </br>
The py script 'facial_data_2_numpy.py' is used to store data of face. </br>
The script reads webcam frames and detects faces using HaaR Cascade Dataset. </br>
Then, a section of that whole frame is taken out in variable 'face_selection' which crops out the face with 10px padding at all sies </br>
This 'face_selection' is then flatten and stored into numpy array in './data/' path with filename as person's name. </br>
The file 'face_classifier.py' detects faces , then with KNN, it classifies its id (0/1/2/3) </br> 
And then using dictionary mapping that that id is linked to that person's name (0->A , 1->B , 2->C......) </br>
A rectangle is formed around the faces and then name (A,B,C) is displayed. </br> 
