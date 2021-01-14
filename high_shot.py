import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def high_shot(img_target,img_mould):#img_target opencv

    h0,w0 = img_target.shape[:2]
    rightdown = [w0,h0]
    leftdown=[0,h0]
    rightup = [w0,0]
    leftup =  [0,0]
        
    h,w = img_mould.shape[:2]  
    pts1 = np.float32([leftdown, rightdown, rightup, leftup])
    pts2 = np.float32([[0,h],[w,h],[w,0],[0,0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    a1_image_all = cv2.warpPerspective(img_target, M, (w, h))
    
    gray = cv2.cvtColor(img_mould, cv2.COLOR_BGR2GRAY)
    res1, a2_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)            
   
    a1_image_all = cv2.cvtColor(a1_image_all, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("result_gray.jpg",a1_image_all)

    a1_image_all = a1_image_all/255.0  #注意255.0得采用浮点数
    img_gamma = np.power(a1_image_all,0.8)
    img_gamma = np.power(img_gamma,0.7)*255.0
    img_gamma[img_gamma > 255] = 255
    # img_gamma = np.power(img_gamma,1.1)*255.0
    a1_image_all = img_gamma.astype(np.uint8)
    cv2.imwrite("result_gamma.jpg",a1_image_all)
    
    a1_image_all = cv2.bilateralFilter(a1_image_all,9,9,1)#双边滤波
    # a1_image_all = max_min_value_filter(a1_image_all,3)#最大值滤波
    a1_image_all = cv2.bilateralFilter(a1_image_all,9,90,1)
    #a1_image_all = cv2.bilateralFilter(a1_image_all,9,9,180)

    cv2.imwrite("result_max_2.jpg",a1_image_all)
    

    
    kernel0 = np.ones((3,3),np.uint8)
    kernel = np.ones((3,3),np.uint8)
    kernel1 = np.ones((2,2),np.uint8)
    
    a1_image_all = cv2.morphologyEx(a1_image_all, cv2.MORPH_CLOSE, kernel0)
    # a1_image = cv2.erode(a1_image, kernel1, iterations =1)
    #a1_image = cv2.bilateralFilter(a1_image,9,9,180)
    a1_image_all = cv2.GaussianBlur(a1_image_all, (11,11), 0)
    cv2.imwrite("result1.jpg",a1_image_all)

    a1_rot = np.rot90(np.rot90(a1_image_all))
    a = direc3(a1_image_all,a2_image)#a为1即同一方向，0则相反
    b = direc3(a1_rot,a2_image)
    #print([a,b])
    if a < b:
        q = 1
        a1_image = a1_rot
    else:
        q=0


    if q==1:
        a1_image_all = np.rot90(np.rot90(a1_image_all))
    else:
        pass
    
    
    h0,w0 = a1_image_all.shape[:2]
    border = 17
    qq = 10
    
    norm_img_left = a1_image_all[:,0:int(w0/2)]
    norm_img_right = a1_image_all[:,int(w0/2):int(w0)]
    norm_img_left_0 = cv2.adaptiveThreshold(norm_img_left,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,border,qq)
    norm_img_right_0 = cv2.adaptiveThreshold(norm_img_right,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,border,qq)    
        #cv2.imwrite("result_3.jpg",norm_img_right_0) 
    img_2 = np.hstack([norm_img_left_0,norm_img_right_0])

    img_2 = cv2.GaussianBlur(img_2, (11,11), 0)
    h1,w1 = img_2.shape[:2]
    img_2[:, 0:int(w1/256)] = 255
    img_2[:, int(w1*255/256):int(w1)] = 255
    img_2[0:int(h1/256), :] = 255
    img_2[int(h1*255/256):int(h1), :] = 255
    cv2.imwrite("result0.jpg",img_2)
    #a1_image = cv2.dilate(a1_image, kernel1, iterations = 2)
    return img_2#二值化图像

def direc3(a1_image,a2_image):
    h , w = a1_image.shape[:2]
    threshold = 0.6
    sum0 = 0
    n = 10
    m = 7
    for i in range(n):#y
        for j in range(m):#x
            if i>0 and j>0 and i<n and j<m:
                square1= a1_image[int((i-1)*h/n) : int((i+2)*h/n) , int((j-1)*w/m) : int((j+2)*w/m)]#裁剪为[y:y0,x:x0]
            else:
                square1= a1_image[int(i*h/n) : int((i+1)*h/n) , int(j*w/m) : int((j+1)*w/m)]
            
            square2= a2_image[int(i*h/n) : int((i+1)*h/n) , int(j*w/m) : int((j+1)*w/m)]
            res = cv2.matchTemplate(square1,square2,cv2.TM_CCOEFF_NORMED)#square1,原图
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            #print(max_val)
            if max_val<threshold:
                pass
            else:
                sum0 +=1
            #print(sum0)
    return sum0
    
