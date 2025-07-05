import cv2
import dlib
import numpy as np

def extract_mouth_roi(video_path, predictor_path, output_size=(96, 96)):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    cap = cv2.VideoCapture(video_path)
    mouth_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            shape = predictor(gray, faces[0])
            # Mouth landmarks: 48-67
            mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            x, y, w, h = cv2.boundingRect(mouth_points)
            mouth_roi = frame[y:y+h, x:x+w]
            mouth_roi = cv2.resize(mouth_roi, output_size)
            mouth_frames.append(mouth_roi)
        else:
            # If no face detected, append a black frame
            mouth_frames.append(np.zeros((*output_size, 3), dtype=np.uint8))
    cap.release()
    return mouth_frames 