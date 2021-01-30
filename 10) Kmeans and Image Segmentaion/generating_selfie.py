import cv2
import numpy as np

# initialize camera where 0 defines the id of webcam
cap = cv2.VideoCapture(0)

# haarcascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# for storing every 10 face we will keep a counter
skip = 0
# for storing every 10 face
face_data = []
# this is the folder where I want to store the dataset
dataset_path = './data/'

# ask the name
file_name = input('enter the name of the person : ')

# this loop will stop only when user presses 'q'
while True:
	# reading what information we are getting through web cam
	ret,frame = cap.read()

	if ret == False:
		continue

	# I want to store in gray frame   # takes frame and color mode # this converts colored RGB frame into grayframe	
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# (list of faces and each face is a typle of x, y, w, h)	# (scaling parameters  # no of neighbours)
	faces = face_cascade.detectMultiScale(frame, 1.5, 5)

	if len(faces) == 0:
		continue
	# sorting the faces wrt to area of faces # area = f[2]*f[3] # we are going to store only the largest face
	faces = sorted(faces, key = lambda f:f[2]*f[3])
	# started from the last face ie -1 because the last face is largest
	for face in faces[-1:]:
		# (x,y) are cordinates and (w and h) width and height of face 
		x,y,w,h = face
		# face bounding                         # color = mixture of green and red
		cv2.rectangle(frame, (x,y),(x+w, y+h), (0, 255, 255), 2)
		# extract (crop out the required face): region of interest	
		offset = 10 # pixel
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		skip += 1
		# for storing every 10 face
		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow('Frame', gray_frame)
	#cv2.imshow('Face Section', face_section)


	#8 bit integer  # 32 bit integer  &  # 8 bit integer 
	key_pressed = cv2.waitKey(1) & 0xFF 

	# key pressed == ascii value of q
	if key_pressed == ord('q'):
		break
	# convert our face list array into a numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# save the data in the path
np.save(dataset_path+file_name+'.npy', face_data)
print('data sucessfully saved at ' + dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()