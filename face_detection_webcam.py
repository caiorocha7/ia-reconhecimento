import cv2
import numpy as np
import os

# the same function we've used before
from helper_functions import resize_video

### Choose the face detector
# -> Options: haarcascade  |  ssd
detector = "ssd"  # use SSD if you want more accurate detections
max_width = 800           # leave None if you don't want to resize and want to keep the original size of the video stream frame



# The return of the following two functions were constructed on the assumption that there's only one face appearing during the video stream.
# That's the objective, capture one face at a time.

# Return the detected face using Haar cascades
def detect_face(face_detector, orig_frame):
    frame = orig_frame.copy()  # to keep the original frame intact (just if we want to save the full image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces and its coordinates
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    # x and y coordinates, width and height
    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return frame

# Return the detected face using SSD
def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy()  # to keep the original frame intact (just if we want to save the full image
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

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            if show_conf:
                text_conf = "{:.2f}%".format(confidence * 100)
                cv2.putText(frame, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return frame

if detector == "haarcascade":
    # For Face Detection with HAAR CASCADE -> import haar cascade for face detection
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
else:
    # For Face Detection with SSD (OpenCV's DNN) -> load weights from caffemodel
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

    if detector == "haarcascade":
        processed_frame = detect_face(face_detector, frame)
    else:  # SSD
        processed_frame = detect_face_ssd(network, frame)

    cv2.imshow("Detecting faces", processed_frame)
    cv2.waitKey(1)

print ("Finished!")
cam.release()
cv2.destroyAllWindows()