# coding: utf-8
import sys, cv2, time, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fastapi import FastAPI
import json
import requests
import pickle
import math
import numpy
import base64    



# _BASEDIR = os.path.dirname(os.path.abspath(__file__))
# os.chdir(_BASEDIR)

from flask import request, Flask

from io import StringIO
import matplotlib.pyplot as plt
from io import BytesIO

import base64

server = Flask(__name__)


def high_shot(img_target):#img_target opencv
    #img_2 = cv2.imread(img_target)  # 高拍仪图片，有黑边
    #print(img_target)

   
    h0,w0 = img_target.shape[:2]
    #h0,w0 = img_target.shape[:2]
    list1 = line_detect_possible_demo(img_target)
    if type(list1) == str:
        return list1 
    else:
        list3 = []
        list_4 = []
        list0 = list1[0]
        xx = list1[1]
        yy = list1[2]
        print([xx,yy])
        list_k1=[]
        list_k2= []
        #list0 = line_detect_possible_demo(img_2)
        for i in range(len(list0)):
            for j in range(len(list0)):
                if list0[i][1]!=list0[i][0] and list0[j][1]!=list0[j][0]:
                    #print("1111111")
                    k1 = (list0[i][3]-list0[i][2])/(float(list0[i][1]-list0[i][0]))
                    k2 = (list0[j][3]-list0[j][2])/(float(list0[j][1]-list0[j][0]))
                    try:                  
                        cobb =float(math.fabs(np.arctan((k1-k2)/(float(1 + k1*k2)))*180/np.pi)+0.5)#k1*k2+1= 0
                    except RuntimeWarning:
                        cobb = 90
                    #print(cobb)
                    if 85 <cobb <95  :
                        k1_0 = k1
                        k2_0 = k2
                        x, y = cross_point(list0[i], list0[j])
                        list3.append([x, y])
                        list_k1.append([k1_0,k2_0])
                    elif abs(cobb) < 5 and abs(k1) < 0.0875 and abs(k2) <0.0875:
                        k3_0 = k1
                        k4_0 = k2 
                        x =  list0[i][0]   
                        y = list0[i][2] 
                        x1 =  list0[i][1]   
                        y1 = list0[i][3]   
                        list_4.append([x, y])
                        list_4.append([x1, y1])
                        x2 =  list0[j][0]   
                        y2 = list0[j][2] 
                        x3 =  list0[j][1]   
                        y3 = list0[j][3]   
                        list_4.append([x2, y2])
                        list_4.append([x3, y3])
                        list_k2.append([k3_0,k4_0]) 
                        #print(list3)
                        #print("11111111111")       
                    else:
                        k1 =  "未找到纸张"
                        k2 =  "未找到纸张"
                        list_k1.append([k1,k2])
                elif list0[i][1]==list0[i][0] and list0[j][1]!=list0[j][0]:
               
                    k2 = (list0[j][3]-list0[j][2])/(float(list0[j][1]-list0[j][0]))
                    if -5<(math.atan(abs(k2))*180/np.pi) <5:
                        k1_0 =  "无穷大"
                        k2_0 = k2
                        x, y = cross_point(list0[i], list0[j])
                        list3.append([x, y])
                        list_k1.append([k1_0,k2_0])
                elif list0[i][1]!=list0[i][0] and list0[j][1]==list0[j][0]: 
                    
                    k1 = (list0[i][3]-list0[i][2])/(float(list0[i][1]-list0[i][0]))
                    if -5<(math.atan(abs(k1))*180/np.pi) <5:
                        k2_0 = k1
                        k1_0 = "无穷大"
                        x, y = cross_point(list0[i], list0[j])
                        list3.append([x, y])
                        list_k1.append([k1_0,k2_0])
                else:
                    #k1 = (list0[j][3]-list0[j][2])/(float(list0[j][1]-list0[j][0]))   
                    #if -5<(math.atan(abs(k1))*180/np.pi) <5:
                    #    k2_0 = k1
                    #    k1_0 = "无穷大"
                    #    x, y = cross_point(list0[i], list0[j])
                    #    list3.append([x, y])
                    #    list_k1.append([k1_0,k2_0])
                    k1_0 = "无穷大"
                    k2_0 = 0
                    list_k1.append([k1_0,k2_0])
                    x, y = cross_point(list0[i], list0[j])
        #print([k1_0,k2_0])
        if all(list_k1[i][0]) == "未找到纸张" for i in range(len(list_k1)):
            return   "未找到纸张"
        elif  all(abs(list_k2[i][0])  <= 0.0875 and abs(list_k2[i][1])  <= 0.0875  for i in range(len(list_k2))) and list3 == []:
            #print("22222")
            #print(list3)
            list8 =list_4                                                                                                                                                                                                                                                                                                                                                                                                   
            list4 = []
            list4_0 = []
            list4_1 = []
            list4_2 = []
            list4_3 = [] 
            #print(list8)
            #print([xx,yy])
            for i in range(len(list8)):  
                if  list8[i][1]>yy and list8[i][0] < xx:
                    #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < y/2:
                    list4_0.append([list8[i][0], list8[i][1]])
            
            
                elif list8[i][1]<yy and list8[i][0] < xx:
                    #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < h0/2:
                    list4_1.append([list8[i][0], list8[i][1]])
            
            
                elif  list8[i][1] > yy and list8[i][0]> xx:
                    list4_2.append([list8[i][0], list8[i][1]])
                
                
                elif list8[i][1]< yy and list8[i][0] >xx:
                    #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) <h0/2:
                    list4_3.append([list8[i][0], list8[i][1]])               
                else: 
                    pass   
        else:                      
            list8 = list(filter(lambda x: x != [-1, -1], list3))          
            listk = list(filter(lambda x: x[0] != "未找到纸张" and x[1]!="未找到纸张" , list_k1))  
            k1_0 = listk[0][0] 
            k2_0 = listk[0][1]                                                                                                                                                                                                                                                                                                                                                                                          
            list4 = []
            list4_0 = []
            list4_1 = []
            list4_2 = []
            list4_3 = []
            if type(k1_0) != str and type(k2_0) != str:
                if k1_0 > 0 and k2_0<0:
                    kx = k1_0
                    ky = k2_0
                    #print("111111111")
                    for i in range(len(list8)):   
                        #print([xx,yy])
                        if  list8[i][1]-(ky*list8[i][0]+yy-ky*xx)  >0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) > 0:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < y/2:
                            list4_0.append([list8[i][0], list8[i][1]])
                    
                    
                        elif list8[i][1]-(ky*list8[i][0]+yy-ky*xx) < 0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) > 0:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < h0/2:
                            list4_1.append([list8[i][0], list8[i][1]])
                    
                    
                        elif  list8[i][1]-(ky*list8[i][0]+yy-ky*xx) > 0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) < 0:
                            list4_2.append([list8[i][0], list8[i][1]])
                        
                        
                        elif list8[i][1]-(ky*list8[i][0]+yy-ky*xx) < 0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) < 0:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) <h0/2:
                            list4_3.append([list8[i][0], list8[i][1]])               
                        else:
                            pass
                elif k1_0 <0 and k2_0>0:
                    kx = k2_0
                    ky = k1_0
            
                    for i in range(len(list8)):   
                        if  list8[i][1]-(ky*list8[i][0]+yy-ky*xx)  >0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) > 0:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < y/2:
                            list4_0.append([list8[i][0], list8[i][1]])
                    
                    
                        elif list8[i][1]-(ky*list8[i][0]+yy-ky*xx) < 0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) > 0:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < h0/2:
                            list4_1.append([list8[i][0], list8[i][1]])
                    
                    
                        elif  list8[i][1]-(ky*list8[i][0]+yy-ky*xx) > 0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) < 0:
                            list4_2.append([list8[i][0], list8[i][1]])
                        
                        
                        elif list8[i][1]-(ky*list8[i][0]+yy-ky*xx) < 0 and list8[i][1]-(kx*list8[i][0]+yy-kx*xx) < 0:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) <h0/2:
                            list4_3.append([list8[i][0], list8[i][1]])               
                        else:
                            pass
                else:
                    
                    for i in range(len(list8)):  
                        if  list8[i][1]>yy and list8[i][0] < xx:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < y/2:
                            list4_0.append([list8[i][0], list8[i][1]])
                    
                    
                        elif list8[i][1]<yy and list8[i][0] < xx:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < h0/2:
                            list4_1.append([list8[i][0], list8[i][1]])
                    
                    
                        elif  list8[i][1] > yy and list8[i][0]> xx:
                            list4_2.append([list8[i][0], list8[i][1]])
                        
                        
                        elif list8[i][1]< yy and list8[i][0] >xx:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) <h0/2:
                            list4_3.append([list8[i][0], list8[i][1]])               
                        else:
                            pass
            else:

                if type(k1_0) == str and type(k2_0)!=str:
                    ky = k2_0
                    for i in range(len(list8)):  
                        if   list8[i][1]>yy and list8[i][0] < xx:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < y/2:
                            list4_0.append([list8[i][0], list8[i][1]])
                    
                    
                        elif list8[i][1]<yy and list8[i][0] < xx:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) < h0/2:
                            list4_1.append([list8[i][0], list8[i][1]])
                    
                    
                        elif  list8[i][1]>yy and list8[i][0]> xx:
                            list4_2.append([list8[i][0], list8[i][1]])
                        
                        
                        elif list8[i][1]<yy and list8[i][0] >xx:
                            #if abs(list8[i][0]-list8[j][0]) < w0/2 and abs(list8[i][1]-list8[j][1]) <h0/2:
                            list4_3.append([list8[i][0], list8[i][1]])               
                        else:
                            pass

            # assert list4_0 != [],"文件不平整或破损" 
            # assert list4_1 != [],"文件不平整或破损"
            # assert list4_2 != [],"文件不平整或破损"
            # assert list4_3 != [],"文件不平整或破损"
        if list4_0 == [] or list4_1 ==[] or list4_2==[] or list4_3 ==[]:
            return "文件不平整,破损或未放入"
        else:
            list4.append(list4_0)
            list4.append(list4_1)
            list4.append(list4_2)
            list4.append(list4_3)

            list7 = []
            list9 = []
            dic6_0 = {}
            dic6_1 = {}
            dic6_2 = {}
            dic6_3 = {}
            list9.append(dic6_0)
            list9.append(dic6_1)
            list9.append(dic6_2)
            list9.append(dic6_3)
            #print(list4)   

            for i in range(len(list4)):
                list5 = list4[i]
                dic6 = list9[i]
                for j in range(len(list5)):
                    a = (list5[j][0]-xx)**2 + (list5[j][1] - yy)**2
                    dic6[float(a)] = [list5[j][0], list5[j][1]]
                assert dic6.keys() != None,0
                b = min(dic6.keys())
                
                list7.append(dic6[b]) 
            #print(list7)
                

            rightdown = list7[2]
            leftdown= list7[0]
            rightup = list7[3]
            leftup = list7[1]


            print(list7)
            h = int(math.sqrt((leftup[0]-leftdown[0])**2+(leftup[1]-leftdown[1])**2))
            w = int(math.sqrt((leftup[0]-rightup[0])**2+(leftup[1]-rightup[1])**2))
            print([h,w])
            if h>w:
                pass
            else:
                rightdown= list7[0]
                rightup= list7[2]
                leftdown = list7[1]
                leftup = list7[3]
                h = int(math.sqrt((leftup[0]-leftdown[0])**2+(leftup[1]-leftdown[1])**2))
                w = int(math.sqrt((leftup[0]-rightup[0])**2+(leftup[1]-rightup[1])**2))

            print([leftup,rightup,leftdown,rightdown])
            pts1 = np.float32([leftdown, rightdown, rightup, leftup])  # 变换前四点
            pts2 = np.float32([[0, h], [w, h], [w, 0], [0, 0]])  # 变换后四点
            M = cv2.getPerspectiveTransform(pts1, pts2)
            a1_image = cv2.warpPerspective(img_target, M, (w, h)) 
            print("high_shot complete") 
                
            cv2.imwrite("高拍仪a1_image.jpg", a1_image)
        
            a1_image[:, 0:int(w/256)] = 255
            a1_image[:, int(w*255/256):int(w)] = 255
            a1_image[0:int(h/256), :] = 255
            a1_image[int(h*255/256):int(h), :] = 255
            # cv2.imwrite("高拍仪a1_threshold_high.jpg", a1_image)
            
            
            cv2.imwrite("result.jpg", a1_image)
        
            return a1_image
        
       
    



def cross_point(a, b):  # 计算交点函数 保留
    x1 = a[0]  # 取四点坐标    
    x2 = a[1]
    y1 = a[2]
    y2 = a[3]

    x3 = b[0]
    x4 = b[1]
    y3 = b[2]
    y4 = b[3]
    if x1 != x2:
        k1 = (y2-y1)*1.0/(x2-x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1*1.0-x1*k1*1.0  # 整型转浮点型是关键
        if (x4-x3) == 0:  # L2直线斜率不存在操作
            k2 = None
            b2 = 0
        else:
            k2 = (y4-y3)*1.0/(x4-x3)  # 斜率存在操作
            b2 = y3*1.0-x3*k2*1.0
        if k2 == None:
            x = x3
            y = k1*x*1.0+b1*1.0
        else:
            if k1 != k2:
                x = (b2-b1)*1.0/(k1-k2)
                y = k1*x*1.0+b1*1.0
            else:
                x = -1
                y = -1

    else:
        if (x4-x3) == 0:  # L2直线斜率不存在操作
            k1 = None
            b2 = 0
            x = -1
            y = -1

        else:
            k2 = (y4-y3)*1.0/(x4-x3)  # 斜率存在操作
            b2 = y3*1.0-x3*k2*1.0
            x = x1
            y = k2*x*1.0+b2*1.0

    return x, y

def line_detect_possible_demo(image):#保留

    list0 = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    res1, th_img3 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("th_img3.jpg", th_img3)
    h_0,w_0 = th_img3.shape[:2]
   
    # apertureSize是sobel算子大小，只能为1,3,5，7越小越好
    # edges = cv2.Canny(th_img3, 80,115, apertureSize=3)
    # cv2.imwrite("2_3.jpg", edges)
    contours, hierarchy = cv2.findContours(
        th_img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(len(contours[0]))
    h_temp,w_temp = th_img3.shape[:2] 
    #print(w_temp)

    for i in range(len(contours)):
        for j in range(len(contours[i])):
            if contours[i][j][0][1] < (h_0*4/5):
                m = i
                break
            else:
                m=-1
    if m ==-1:
        return "未找到文件"
    else:
        sum0 = 0
        sum1 = 0
        x_max = 0
        x_min = w_0
        y_max = contours[m][0][0][1]
        y_min = contours[m][0][0][1]
        for i in range(len(contours[m])):        
            sum0 += contours[m][i][0][1]    
            sum1 += contours[m][i][0][0]
            if x_max < contours[m][i][0][0]: 
                x_max = contours[m][i][0][0]
            if contours[m][i][0][0]<x_min:
                x_min = contours[m][i][0][0]
            if y_max < contours[m][i][0][1]: 
                y_max = contours[m][i][0][1] 
            if y_min > contours[m][i][0][1]: 
                y_min = contours[m][i][0][1] 
        print([x_min,x_max])
        print([y_min,y_max])
        
        yy = sum0/len(contours[m])
        xx = sum1/len(contours[m])
        print([xx,yy])
        x_dis = x_max - x_min

        if y_max>h_0*31/32 or x_min< (w_0/32) or x_max>w_0*31/32 or y_min< (h_0/32):
            return "位置异常,请重新放置"
        else:
            if x_dis >w_0*31/32:
                x_max = 0
                x_min = w_0
                for i in range(len(contours[m])):
                    if contours[m][i][0][1]  < yy:           
                        if x_max < contours[m][i][0][0]: 
                            x_max = contours[m][i][0][0]
                        if contours[m][i][0][0]<x_min:
                            x_min = contours[m][i][0][0]
            #print("len of x")
            #print([x_min,x_max])
            #print([y_min,y_max])        


            x_dis = x_max - x_min  
            y_dis = y_max - y_min
            if x_dis >= y_dis:
                h_temp = y_dis/2
            else:
                h_temp = x_dis/2
            # drawSrc=edges.copy()#注意需要copy，否则原图会发生改变
            drawSrc = np.zeros((h_0, w_0, 3), np.uint8)
            res = cv2.drawContours(drawSrc, contours, m, (255, 255, 255), 5)
            cv2.imwrite("2_4.jpg", res)

            # 转成单通道
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            res1, res2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#倒数第三个参数默认为-1，表示绘制全部轮廓，
            #指定为其他值，则在图像中进行选择单个，并且有内外圈之分；最后一个参数为颜色宽度
            cv2.imwrite("res2.jpg", res2)

            # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
            #lines = cv2.HoughLines(res2, 1, np.pi/180, 220)
            
            lines = cv2.HoughLinesP(res2,1, np.pi/180,300, minLineLength=h_temp/5,maxLineGap=100)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线 大于maxlinegap会将两条直线看成一条
            
            if type(lines) != numpy.ndarray:
                return "未找到纸张"
            else:
                print(len(lines))  #300 /5 300 OK 50             300 /5 200 OK 46   300 /5 150 OK 41        300 /5 150 OK 37
                if len(lines) > 45:
                    print("再次循环")
                    lines = cv2.HoughLinesP(res2,1, np.pi/180,400, minLineLength=h_temp/3.5,maxLineGap=300) 
                else:
                    pass
                print("拟合直线数:"+str(len(lines)))
                #print("拟合直线数:"+str(len(lines)))
                lines1 = lines[:, 0, :] 
                for x1,y1,x2,y2 in lines1[:]:
                    list0.append([x1, x2, y1, y2])
                list1= [list0,xx,yy]
                return list1
    
# def image_to_base64(image_np):
 
#     image = cv2.imencode('.jpg',image_np)[1]
#     image_code = str(base64.b64encode(image))[2:-1]
 
#     return image_code





@server.route("/", methods=['POST'])
def get_frame():
   
    start_time = time.time()
    upload_file = request.get_data()
    req = json.loads(upload_file)
    #old_file_name = upload_file.filename
    
    if upload_file:
        #name = req['name']
        #print(name)
        
        img_str = req['image_0'] #得到unicode的字符串
        img_decode_ = img_str.encode('ascii') #从unicode变成ascii编码
        img_decode = base64.b64decode(img_decode_) #解base64编码，得图片的二进制
        #img_np_ = np.frombuffer(img_decode, np.uint8)
        #ori_image = base64.b64decode(img_decode)
        fout = open("img_base64.jpg","wb")
        fout.write(img_decode)
        fout.close
        img = cv2.imread("img_base64.jpg")
        #img = cv2.imdecode(img_np_, cv2.COLOR_RGB2BGR) #转为opencv格式
        
        a1_0=high_shot(img)
        if type(a1_0) == str:
            return a1_0
        else:
            cv2.imwrite("a1_0.jpg",a1_0)
            #img_code = image_to_base64(a1_0)
            #with open("a1_0.jpg","rb") as f: 
            f=open("a1_0.jpg", "rb") 
            base64_data = base64.b64encode(f.read())
            f.close() 
            return base64_data
    else:    
        return false
    


# if __name__ == "__main__":

#     server.run("192.168.2.236", port=6060)


    

              










