import cv2
import numpy as np
import os
import pickle
import re
import face_recognition

# the same function we've used before
from helper_functions import resize_video

pickle_name = "face_encodings_custom.pickle" # the name of the pickle file where are stored the encodings
max_width = 800           # leave None if you don't want to resize and want to keep the original size of the video stream frame (but will take longer to process)

# Load encodings from pickle file 
data_encoding = pickle.loads(open(pickle_name, "rb").read())
list_encodings = data_encoding["encodings"]
list_names = data_encoding["names"]

# Recognize the faces in a given image (in this case, the frame of the video)
# the function will return: face locations, face names, confidence values
def recognize_faces(image, list_encodings, list_names, resizing=0.25, tolerance=0.6):
  image = cv2.resize(image, (0, 0), fx=resizing, fy=resizing)

  img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  face_locations = face_recognition.face_locations(img_rgb)
  face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

  face_names = []     # list used to store pred names results
  conf_values = []    # list used to store pred conf results
  for encoding in face_encodings:
    # see if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(list_encodings, encoding, tolerance=tolerance)
    name = "Not identified"
    
    # use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(list_encodings, encoding)
    best_match_index = np.argmin(face_distances)  # get index of the lower distance
    if matches[best_match_index]:
      name = list_names[best_match_index] # get name from list_names 
    face_names.append(name) 
    conf_values.append(face_distances[best_match_index])
    #print(best_match_index, matches, face_names)

  # convert to numpy array to adjust coordinates with frame resizing 
  face_locations = np.array(face_locations)
  face_locations = face_locations / resizing
  return face_locations.astype(int), face_names, conf_values

# show the recognition over the image (using the returned face locations, face names, confidence values) 
def show_recognition(frame, face_locations, face_names, conf_values):

  # "unzip" the parameters of the face (face locations, face names, confidence values) 
  for face_loc, name, conf in zip(face_locations, face_names, conf_values):
    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]  # get the coordinates of the bounding box (ROI of the face)

    conf = "{:.8f}".format(conf)
    cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 255, 0), 4)   
    if name is not "Not identified": 
        cv2.putText(frame, conf,(x1, y2 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 255, 0), 1, lineType=cv2.LINE_AA)

  return frame


# video capture object (webcam) 
cam = cv2.VideoCapture(0)


# loop over every frame of the video stream
while(True):
    ret, frame = cam.read()

    # resize only if a max_width is specified
    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    face_locations, face_names, conf_values = recognize_faces(frame, list_encodings, list_names, 0.25)
    print(face_locations)
    processed_frame = show_recognition(frame, face_locations, face_names, conf_values)

    cv2.imshow("Recognizing faces", frame)
    cv2.waitKey(1)

print ("Completed!")
cam.release()
cv2.destroyAllWindows()