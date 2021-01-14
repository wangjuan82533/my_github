# coding: utf-8
import sys, cv2, time, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fastapi import FastAPI
import json
import requests
import pickle
from flask import request, Flask
from io import StringIO
from io import BytesIO
import base64

from high_shot import *
from pic_matchtemp import*



server = Flask(__name__)

@server.route("/", methods=['POST'])
def get_frame():
   
    start_time = time.time()
    upload_file = request.get_data()
    req = json.loads(upload_file)
    #old_file_name = upload_file.filename
    
    if upload_file:
        start=time.time()
        img_str = req['image_0'] #得到unicode的字符串
        img_decode_ = img_str.encode('ascii') #从unicode变成ascii编码
        img_decode = base64.b64decode(img_decode_) #解base64编码，得图片的二进制
        img_np_ = np.frombuffer(img_decode, np.uint8)
        img = cv2.imdecode(img_np_, cv2.COLOR_RGB2BGR) #转为opencv格式
        img_str_1 = req['image_1'] #得到unicode的字符串
        img_decode_1 = img_str_1.encode('ascii') #从unicode变成ascii编码
        img_decode_1 = base64.b64decode(img_decode_1) #解base64编码，得图片的二进制
        img_np_1 = np.frombuffer(img_decode_1, np.uint8)
        img_1 = cv2.imdecode(img_np_1, cv2.COLOR_RGB2BGR) #转为opencv格式
        n = 10
        
        
        #print("start"+"===============================================================================")
        a1_0= high_shot(img,img_1)
        #img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("resul1.jpg",img_1)
        img_1 = cv2.bilateralFilter(img_1,9,200,1) # lastone bigger  越浅
        #img_1 = cv2.GaussianBlur(img_1, (15,15), 0)
        cv2.imwrite("resul2_0.jpg",img_1)
       # img_1 = cv2.adaptiveThreshold(img_1,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,11)
     #img_1 = cv2.Canny(img_1, 50, 150)
        # kernel = np.ones((1,1),np.uint8)
        kernel0 = np.ones((3,3),np.uint8)
        # kernel1 = np.ones((1,1),np.uint8)
        # img_1 = cv2.dilate(img_1, kernel, iterations = 2)
        img_1 = cv2.morphologyEx(img_1, cv2.MORPH_CLOSE, kernel0)
        img_1 = cv2.GaussianBlur(img_1, (15,15), 0)
        img_1 = cv2.adaptiveThreshold(img_1,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,8.3)
        # img_1 = cv2.erode(img_1, kernel1, iterations = 1)
        img_1 = cv2.GaussianBlur(img_1, (11,11), 0)
        h0,w0 = img_1.shape[:2]
        img_1[:, 0:int(w0/256)] = 255
        img_1[:, int(w0*255/256):int(w0)] = 255
        img_1[0:int(h0/256), :] = 255
        img_1[int(h0*255/256):int(h0), :] = 255
    
        cv2.imwrite("resul.jpg",img_1)
        if type(a1_0) == str:
            return a1_0
        else:              
            similar0,list_max, list_0= direc2(a1_0,img_1,n)
            a = b = c = d = e= f = g =0
            average = sum(list_max)/len(list_max)
           # print(average)
           # print(list_max)
            for i in range(len(list_max)):
                if list_max[i]>= 0.85:
                    a += 1               
                elif list_max[i]>=0.75:
                    b +=1
                elif list_max[i] >=0.48:
                    c+=1
                elif  list_max[i] >= 0.4:
                    d +=1
                elif list_max[i]>=0.25:
                    e+=1
                elif list_max[i]>= 0:
                    f+=1
                else:
                    g+=1
            
            print([a,b,c,d,e,f,g])
            var = [a,b,c,d,e,f,g]

            if (a+b+c >55) :
                m = 1
            elif 40<(a+b+c )<=55:
                m = 0.5
            else:
                m = 0
        
            # print(str(time.time()-start))  
            return str(m)
            
    else:
        return "false" 


if __name__ == "__main__":

    server.run("192.168.2.226", port=8080)


    

              









