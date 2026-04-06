import cv2
import face_recognition

known_faces = []
known_names = ["Aprajita", "Shivank"]
image_files = ["Aprajita.jpg", "Shivank.jpg"]

for file in image_files:
    img = face_recognition.load_image_file(file)
    encoding = face_recognition.face_encodings(img)[0]
    known_faces.append(encoding)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        results = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Access Denied"
        color = (0, 0, 255)

        if True in results:
            first_match_index = results.index(True)
            name = known_names[first_match_index]
            color = (0, 255, 0)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()