import cv2
import numpy as np
import dlib
import os

def brute_force_match(iris_1, iris_2):

    orb = cv2.ORB_create()
    
    keypoints_img1, des1 = orb.detectAndCompute(iris_1, None)
    keypoints_img2, des2 = orb.detectAndCompute(iris_2, None)

    brute_f = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matchesOriginal = brute_f.match(des1, des1) 
    matchesNew = brute_f.match(des1, des2)   

    if len(matchesOriginal) != 0:
            match_rate = (len(matchesNew)/len(matchesOriginal))*100
    else:
        match_rate = 0
        print("Image Quality is low definition, unable to verify. please use a stronger camera.")

    if match_rate > 35:
            print("IRIS MATCH FOUND IN DATABASE. (match_rate = "+str(match_rate)+")")
            return True
    else:
        print("NO IRIS MATCH FOUND IN DATABASE. (match_rate = "+str(match_rate)+")")
        return False

veri = 0
falsi = 0
for x in range(108):
    
    files = os.listdir('Dataset/CASIA1/'+str(x+1))
    
    if len(files) > 0:

        image_1 = cv2.imread('Dataset/CASIA1/'+str(x+1)+'/'+files[0])
        image_2 = cv2.imread('Dataset/CASIA1/'+str(x+1)+'/'+files[2])
        print(x+1)
        bf_result = brute_force_match(image_1, image_2)
        if (bf_result):
            veri += 1
        else:
            falsi += 1
        
print ("Percentuale di successo: " + str(veri) + " su " + str(veri+falsi)) 


