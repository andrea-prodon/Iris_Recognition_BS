import cv2
import dlib
import numpy as np

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Face Localization/shape_predictor_68_face_landmarks.dat")

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:

        poslandmarkpoints = predictor(gray, face)

        # LEFT EYE

        left_eye1x = poslandmarkpoints.part(38).x
        left_eye1y = poslandmarkpoints.part(38).y
        left_eye2x = poslandmarkpoints.part(41).x
        left_eye2y = poslandmarkpoints.part(41).y
        
        leftEyeTrack = np.array([(left_eye1x-40,left_eye1y-20),
                                 (left_eye1x-40,left_eye2y+20),
                                 (left_eye2x+40,left_eye2y-20),
                                 (left_eye2x+40,left_eye2y+20)
                                 ],
                                np.int32)

        lemin_x = np.min(leftEyeTrack[:, 0])
        lemax_x = np.max(leftEyeTrack[:, 0])
        lemin_y = np.min(leftEyeTrack[:, 1])
        lemax_y = np.max(leftEyeTrack[:, 1])

        left_eye = frame[lemin_y : lemax_y, lemin_x : lemax_x]
        left_eye = cv2.resize(left_eye, None, fx = 5, fy = 5) 
        
        # RIGHT EYE

        right_eye1x = poslandmarkpoints.part(44).x
        right_eye1y = poslandmarkpoints.part(44).y
        right_eye2x = poslandmarkpoints.part(47).x
        right_eye2y = poslandmarkpoints.part(47).y
        
        rightEyeTrack = np.array([(right_eye1x-40,right_eye1y-20),
                                 (right_eye1x-40,right_eye2y+20),
                                 (right_eye2x+40,right_eye2y-20),
                                 (right_eye2x+40,right_eye2y+20)
                                 ],
                                np.int32)

        remin_x = np.min(rightEyeTrack[:, 0])
        remax_x = np.max(rightEyeTrack[:, 0])
        remin_y = np.min(rightEyeTrack[:, 1])
        remax_y = np.max(rightEyeTrack[:, 1])

        right_eye = frame[remin_y : remax_y, remin_x : remax_x]
        right_eye = cv2.resize(right_eye, None, fx = 5, fy = 5) 
        
        
    cv2.imshow("Right Eye", right_eye)
    cv2.imshow("Left Eye", left_eye)

    key = cv2.waitKey(1)
    if key == 27:   
        break

cap.release()
cv2.destroyAllWindows()