import cv2
import numpy as np
import os
from PIL import Image
import pickle


training_path = 'dataset/'

def get_image_data(path_train):
  subdirs = [os.path.join(path_train, f) for f in os.listdir(path_train)]
  #print(subdirs)
  faces = []
  ids = []

  face_names = {}
  id = 1   # current id  (starting id)
  print("Loading faces from training set...")
  for subdir in subdirs:
    name = os.path.split(subdir)[1]

    images_list = [os.path.join(subdir, f) for f in os.listdir(subdir)]
    for path in images_list:
      image = Image.open(path).convert('L')
      face = np.array(image, 'uint8')
      face = cv2.resize(face, (90, 120))
      print(str(id) + " <-- " + path)
      ids.append(id)
      faces.append(face)
      cv2.imshow("Training faces...", face)
      cv2.waitKey(50)

    if not name in face_names:
      face_names[name] = id
      id += 1

  return np.array(ids), faces, face_names

ids, faces, face_names = get_image_data(training_path)

print(ids)
print(len(faces))

print(face_names)

for n in face_names:
  print(str(n) + " => ID " + str(face_names[n]))

# store names and ids in a pickle file
with open("face_names.pickle", "wb") as f:
  pickle.dump(face_names, f)


print('\n')
print('Training Eigenface recognizer......')
eigen_classifier = cv2.face.EigenFaceRecognizer_create()
eigen_classifier.train(faces, ids)
eigen_classifier.write('eigen_classifier.yml')
print('... Completed!\n')

print('Training Fisherface recognizer......')
fisher_classifier = cv2.face.FisherFaceRecognizer_create()
fisher_classifier.train(faces, ids)
fisher_classifier.write('fisher_classifier.yml')
print('... Completed!\n')

print('Training LBPH recognizer......')
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')
print('... Completed!\n')
