import cv2
import numpy as np
import os
from PIL import Image
import pickle
import dlib
import sys
import imutils
import face_recognition


training_path = 'dataset/'                 # path where the face images are located
pickle_filename = "face_encodings_custom.pickle"  # name of the pickle file where we'll be saving the encodings

def load_encodings(path_dataset):
    list_encodings = []
    list_names = []

    # Store image encoding and names
    subdirs = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if os.path.isdir(os.path.join(path_dataset, f))]

    for subdir in subdirs:
        name = subdir.split(os.path.sep)[-1]  # get the name of the subdirectory (which is named after the person)
        images_list = [os.path.join(subdir, f) for f in os.listdir(subdir) if not os.path.basename(f).startswith(".")]

        for image_path in images_list:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            print(name + " <-- " + image_path)

            # Get encoding
            face_roi = face_recognition.face_locations(img, model="cnn")  # cnn or hog

            if len(face_roi) > 0:
                # only to display on cell output
                (start_y, end_x, end_y, start_x) = face_roi[0]
                roi = img[start_y:end_y, start_x:end_x]
                roi = imutils.resize(roi, width=100)
                cv2.imshow('face', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                img_encoding = face_recognition.face_encodings(img, face_roi)
                if len(img_encoding) > 0:
                    # Store file name and file encoding
                    img_encoding = img_encoding[0]
                    list_encodings.append(img_encoding)
                    list_names.append(name)
                else:
                    print("Couldn't encode face from image => {}".format(image_path))  # probably because couldn't find any face on the image
            else:
                print("No face found in image => {}".format(image_path))

    cv2.destroyAllWindows()
    return list_encodings, list_names


list_encodings, list_names = load_encodings(training_path)

print(len(list_encodings))
print(list_names)

# store the encodings and names in a pickle file
encodings_data = {"encodings": list_encodings, "names": list_names}
with open(pickle_filename, "wb") as f:
    pickle.dump(encodings_data, f)

print('\n')
print('Faces encoded successfully!')
