import cv2
import dlib

def show_polyface(frame, poslandmarkpoints):
    #Precision Tracking Of face with full poly fill
    drawlinethickness = 1
    #left eye
    for p in range(36, 41):
        cv2.line(frame, (poslandmarkpoints.part(p).x, poslandmarkpoints.part(p).y), (poslandmarkpoints.part(p+1).x, poslandmarkpoints.part(p+1).y), (0,255,0), drawlinethickness)
    cv2.line(frame, (poslandmarkpoints.part(41).x, poslandmarkpoints.part(41).y), (poslandmarkpoints.part(36).x, poslandmarkpoints.part(36).y), (0,255,0), drawlinethickness) #close off line for left eye
   
   #right eye
    for p in range(42, 47):
        cv2.line(frame, (poslandmarkpoints.part(p).x, poslandmarkpoints.part(p).y), (poslandmarkpoints.part(p+1).x, poslandmarkpoints.part(p+1).y), (0,255,0), drawlinethickness)
    cv2.line(frame, (poslandmarkpoints.part(47).x, poslandmarkpoints.part(47).y), (poslandmarkpoints.part(42).x, poslandmarkpoints.part(42).y), (0,255,0), drawlinethickness) #close off line for right eye

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Face Localization/shape_predictor_68_face_landmarks.dat") #assigned coordinates of the face by DLIB

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        cv2.putText(frame, 'Please Align Face in Center.', (140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    
    for face in faces:
        x_left, y_up  = face.left(), face.top() 
        x_right, y_down = face.right(), face.bottom() 
        cv2.rectangle(frame, (x_left-20,y_up-20), (x_right+20,y_down+20), (0,255,0), 1) 
        poslandmarkpoints = predictor(gray, face)

        show_polyface(frame, poslandmarkpoints)
        
    cv2.imshow("FaceTracking", frame)

    key = cv2.waitKey(1)
    if key == 27:   
        break

cap.release()
cv2.destroyAllWindows()