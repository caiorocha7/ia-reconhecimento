import cv2
import numpy as np
import os
import pickle
import sys

# the same function we've used before
from helper_functions import resize_video

# Face recognizer options ->  eigenfaces  |  fisherfaces  |  lbph (recommended for video)
recognizer = "lbph"
training_data = "lbph_classifier.yml"  # the path to the .yml file
threshold = 10e5   # leave 10e5 if you don't want to specify a threshold. Otherwise, specify the value for threshold
                   # 10e5 = 1000000  (a large number so it will always return a prediction)

max_width = 800           # leave None if you don't want to resize and want to keep the original size of the video stream frame

# Function to load the recognizer depending on the chosen option
def load_recognizer(option, training_data):
    if option == "eigenfaces":
        face_classifier = cv2.face.EigenFaceRecognizer_create()
    elif option == "fisherfaces":
        face_classifier = cv2.face.FisherFaceRecognizer_create()
    elif option == "lbph":
        face_classifier = cv2.face.LBPHFaceRecognizer_create()
    else:
        print("The algorithms available are: Eigenfaces, Fisherfaces and LBPH")
        sys.exit()

    face_classifier.read(training_data) #.yml
    return face_classifier

face_classifier = load_recognizer(recognizer, training_data)

# load names from pickle file
face_names = {}
with open("face_names.pickle", "rb") as f:
    original_labels = pickle.load(f)
    # we invert key and values because it's easier if we access by ID (which is the index)
    face_names = {v: k for k, v in original_labels.items()}


# Return the detected face using SSD
def recognize_faces(network, face_classifier, orig_frame, face_names, threshold, conf_min=0.7):
    frame = orig_frame.copy()  # to keep the original frame intact (just if we want to save the full image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            # sometimes if the face is closer to the edge of the capture area the detector can return negative values, and this will crash the execution.
            # the recommendation is to keep the face on center of the video, but just to guarantee, let's create this condition to prevent the program from crashing
            if (start_x<0 or start_y<0 or end_x > w or end_y > h):
                #print(start_y,end_y,start_x,end_x)
                continue

            face_roi = gray[start_y:end_y,start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120)) ## preferably the same size choosen when training the images (if you're using eigenfaces and fisherfaces it's mandatory that they have the sams size)

            prediction, conf = face_classifier.predict(face_roi)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            pred_name = face_names[prediction] if conf <= threshold else "Not identified"

            text = "{} -> {:.4f}".format(pred_name, conf)
            cv2.putText(frame, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return frame


network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# video capture object
cam = cv2.VideoCapture(0)

# loop over every frame of the video stream
while(True):
    ret, frame = cam.read()

    # resize only if a max_width is specified
    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    processed_frame = recognize_faces(network, face_classifier, frame, face_names, threshold)

    cv2.imshow("Recognizing faces", processed_frame)
    cv2.waitKey(1)

print ("Finished!")
cam.release()
cv2.destroyAllWindows()