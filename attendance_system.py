import cv2
import numpy as np
import os
import pickle
import face_recognition
import csv
from datetime import datetime

# Carregar os encodings de rostos
pickle_name = "face_encodings_custom.pickle"
data_encoding = pickle.loads(open(pickle_name, "rb").read())
list_encodings = data_encoding["encodings"]
list_names = data_encoding["names"]

# Função para redimensionar vídeo
def resize_video(width, height, max_width=600):
    if width > max_width:
        proportion = width / height
        video_width = max_width
        video_height = int(video_width / proportion)
    else:
        video_width = width
        video_height = height
    return video_width, video_height

# Função para reconhecer rostos
def recognize_faces(image, list_encodings, list_names, resizing=0.25, tolerance=0.6):
    image = cv2.resize(image, (0, 0), fx=resizing, fy=resizing)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    face_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(list_encodings, encoding, tolerance=tolerance)
        name = "Not identified"
        face_distances = face_recognition.face_distance(list_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = list_names[best_match_index]
        face_names.append(name)

    face_locations = np.array(face_locations)
    face_locations = face_locations / resizing
    return face_locations.astype(int), face_names

# Função para registrar presença
def mark_attendance(name):
    today_date = datetime.now().strftime('%Y-%m-%d')
    attendance_file = 'attendance.csv'

    # Verificar se o arquivo existe, senão, criar com cabeçalhos
    file_exists = os.path.isfile(attendance_file)
    if not file_exists:
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Datetime'])

    # Ler o arquivo de presença para verificar se a pessoa já está presente hoje
    with open(attendance_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name and row[1].startswith(today_date):
                return  # Pessoa já presente hoje

    # Adicionar nova entrada de presença
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, dt_string])

# Inicializar captura de vídeo
cam = cv2.VideoCapture(0)
max_width = 800

while True:
    ret, frame = cam.read()

    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    face_locations, face_names = recognize_faces(frame, list_encodings, list_names, 0.25)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if name != "Not identified":
            mark_attendance(name)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
