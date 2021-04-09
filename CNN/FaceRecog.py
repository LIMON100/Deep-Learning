# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 08:14:13 2021

@author: limon
"""


import numpy as np
import face_recognition
import cv2
import os

path = "images/"
images = []
clssNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    clssNames.append(os.path.splitext(cl)[0])
    
print(clssNames)



def findEndcodings(images):
    encodeList = []
    
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList


print("Encoding Complete")
encodeListKnown = findEndcodings(images)
        

cap = cv2.VideoCapture(0)

while  True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        #print(faceDis)
        
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = clssNames[matchIndex].upped()
            print(name)
            
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            
            
    
    cv2.imshow('webcam', img)
    cv2.waitKey(1)
    
    

#faceLoc = face_recognition.face_locations(ImgCr7)[0]
#EnCr7 = face_recognition.face_encodings(ImgCr7)[0]
#cv2.rectangle(ImgCr7, (faceLoc[3] ,faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)


#cv2.imshow("Cristiano Ronaldo", ImgCr7)
#cv2.imshow("Kaka", ImgK)
#cv2.waitKey(0)