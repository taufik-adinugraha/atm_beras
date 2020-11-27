import imutils
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import cv2
import os
import dlib
import face_recognition
import time



THRESHOLD = 0.4



# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "caffe/deploy.prototxt"
modelPath = "caffe/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
predictor = dlib.shape_predictor('caffe/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)




print("[INFO] capture image...")
vs = VideoStream(0).start()
time.sleep(3)


# grab the frame
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	(h, w) = frame.shape[:2]

	# load the known faces
	filenames = os.listdir('embeddings')
	knownNames, knownEmbeddings = [], []
	for filename in filenames:
		knownNames.append(filename.split('.')[0])
		knownEmbeddings.append(np.load(f'embeddings/{filename}'))

	if len(knownNames) != 0:

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

		# face detector to localize faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.9:
				# get bounding box
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# face alignment
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				dlibrect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
				face_aligned = fa.align(frame, gray, dlibrect)
				face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
				# face_aligned = face_aligned[64:192,64:192]
				face_aligned = face_aligned[32:224,32:224]
				(fH, fW) = face_aligned.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# encoding
				try:
					face_encoding = face_recognition.face_encodings(face_aligned)[0]
				except:
					continue

				# use the known face with the smallest distance to the new face
				face_distances = face_recognition.face_distance(knownEmbeddings, face_encoding)
				distance = np.min(face_distances)
				if distance < THRESHOLD:
					best_match_index = np.argmin(face_distances)
					name = knownNames[best_match_index]
				else:
					name = 'SIAPA?'

		try:
			print(name, distance)
			# show the output frame
			cv2.imshow("Cropped Face", cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR))
			ref = cv2.imread(f'dataset/{name}.jpg')
			ref = imutils.resize(ref, width=200)
			cv2.imshow("From Database", ref)
			del name, distance, face_aligned, ref
		except:
			cv2.destroyWindow("Cropped Face")
			cv2.destroyWindow("From Database")

	else:
		print('DATABASE KOSONG')

	cv2.imshow("Snapshot", frame)
	key = cv2.waitKey(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break



# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()