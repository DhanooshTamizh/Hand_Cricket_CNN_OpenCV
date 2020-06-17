# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:44:26 2020

@author: HP
"""
from statistics import mode
import time
import random
import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
n=0
l1=[]
s=0
count=0
com=10
score=0
# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}
label=[6,1,2,3,4,5,0]
while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    print(result)
    am=np.argmax(result)
    final=label[am]
    l1.append(final)
    count+=1
    
    if count==81:
        n=mode(l1)
        if n!=com :
        
            if n!=0:
                com=random.choice([0,1,2,3,4,5])
            else:
                com=10
            s+=n
            l1=[]
            count=0
        else:
            print("game over")
            

            l1=[]
            count=0
            s=0
            
            cv2.putText(frame, "Game Over!! ", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.waitKey(2000)
       
    if result[0][0]>0.8:
        val=6
        
    elif result[0][1]>0.8:
        print(1)
        val=1
        #com=random.choice([0,1,2,3,4,5])
        #time.sleep(3)
        #print(com)
        #s=s+str(val)
        #cv2.putText(frame, s, (100, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    elif result[0][2]>0.8:
        print(2)
        val=2
        
    elif result[0][3]>0.8:
        print(3)
        val=3
    elif result[0][4]>0.8:
        print(4)
        val=4
    elif result[0][5]>0.8:
        print(5)
        val=5
    else :
        val=0
       
    # Sorting based on top prediction
    #prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    """if prediction[0][0]=='ONE':
        user=1
        print(user)
        com=random.choice([0,1,2,3,4,5])
        if user==com:
            print("over,score is ",end=" ")
            print(score)
            break
        else:
            score+=user
    if prediction[0][0]=='TWO':
        user=2
        print(user)
        com=random.choice([0,1,2,3,4,5])
        if user==com:
            print("over,score is ",end=" ")
            print(score)
            break
        else:
            score+=user
    if prediction[0][0]=='THREE':
        user=3
        print(user)
        com=random.choice([0,1,2,3,4,5])
        if user==com:
            print("over,score is ",end=" ")
            print(score)
            break
        else:
            score+=user
            
    if prediction[0][0]=='FOUR':
        user=1
        print(user)
        com=random.choice([0,1,2,3,4,5])
        if user==com:
            print("over,score is ",end=" ")
            print(score)
            break
        else:
            score+=user
            
    if prediction[0][0]=='FIVE':
        user=1
        print(user)
        com=random.choice([0,1,2,3,4,5])
        if user==com:
            print("over,score is ",end=" ")
            print(score)
            break
        else:
            score+=user
    """
    
    # Displaying the predictions
    cv2.putText(frame, "user : %s " %str(n), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Bot : %s " %str(com), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)  
    cv2.putText(frame,"player's score %s " %str(s), (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()