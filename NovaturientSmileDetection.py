import cv2
import numpy as np
import os

'''
Usage:
use predict function for predicting the output

Funtion argument:
1) path of the folder that contains the image
2) argument should be of string data type

Return Type:
numpy array
'''

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

predicted=[]
def find(path):
    file=os.listdir(path)
    file=sorted(file,reverse=False)
    for files in file:
        img=cv2.imread(path+files)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces, faceRejectLevels, faceLevelWeights= face_cascade.detectMultiScale3(gray, 1.3, 5, outputRejectLevels=True)
        for(x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            smile, rejectLevels, levelWeights = smile_cascade.detectMultiScale3(roi_gray,minNeighbors=0,minSize=(1, 1),outputRejectLevels=True)
            '''
            threshold value 2.312
            if predicted value lies below threshold mark it as neutral image(0)
            else mark it as smiling image
            if there is no predicted value mark it as neutral image(0)
            '''
            try:
                if(max(levelWeights)[0]>=2.312):
                    predicted.append(max(levelWeights)[0])
                else:
                    predicted.append(0)
            except:
                predicted.append(0)
    return predicted
answer=[]
def predict(path):
    answer=find(path)
    a=np.array(answer)
    #normalizing the values between 0 and 1 and multiplying with 100 to get values over the range of 0 to 100
    a=(a/6)*100
    return a

# Given below a sample for using the model
path='/home/midhilesh/Documents/MLlearning/genki4k/files/'
b=predict(path)
