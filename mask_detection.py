import cv2
import numpy as np
import random

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def draw_rectangle(image, coordinates, label):
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def simulate_temperature():
    return round(random.uniform(36.0, 37.5), 2)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error capturing the video.")
            break

        faces = detect_face(frame)

        for face in faces:
            x, y, w, h = face
            face_roi = frame[y:y + h, x:x + w]

            # Simulate temperature measurement
            temperature = simulate_temperature()

            # Check if the person is wearing a mask or not
            if temperature >= 37.0:
                label = f"No Mask (Temp: {temperature}°C)"
            else:
                label = f"With Mask (Temp: {temperature}°C)"

            draw_rectangle(frame, face, label)

        cv2.imshow('Mask Detection and Temperature Measurement', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
