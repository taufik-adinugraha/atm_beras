import face_recognition
import cv2
import numpy as np
import imutils
from imutils import paths
import argparse
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import dlib
import os
import time



# load our serialized face detector from disk
protoPath = "caffe/deploy.prototxt"
modelPath = "caffe/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
predictor = dlib.shape_predictor('caffe/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

while True:
    time.sleep(30)
    imagePaths_django = list(imutils.paths.list_images('../media/foto'))
    images_django = [i.split('/')[-1] for i in imagePaths_django]
    imagePaths_app = list(imutils.paths.list_images('dataset'))
    images_app = [i.split('/')[-1] for i in imagePaths_app]

    # update database
    for image, imagePath in zip(images_app, imagePaths_app):
        if image in images_django:
            continue
        else:
            name = image.split('.')[0]
            os.system(f'rm embeddings/{name}.*')
            os.system(f'rm dataset/{name}.*')

    for image, imagePath in zip(images_django, imagePaths_django):
        if image in images_app:
            continue
        else:
            # copy image from django database
            os.system(f'cp {imagePath} dataset/{image}')
            # extract the name
            name = image.split('.')[0]

            # load the image and resize it
            image = cv2.imread(f'dataset/{image}')
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize faces
            detector.setInput(imageBlob)
            detections = detector.forward()


            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability)
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.9:

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # face alignment
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dlibrect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    face_aligned = fa.align(image, gray, dlibrect)
                    face_aligned = face_aligned[32:224,32:224]
                    (fH, fW) = face_aligned.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # encoding
                    try:
                        face_encoding = face_recognition.face_encodings(face_aligned)[0]
                        # save
                        np.save(f'embeddings/{name}', face_encoding)
                    except:
                        continue

