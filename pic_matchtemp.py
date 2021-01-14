# coding: utf-8
#import tr
import sys, cv2, time, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from high_shot import*
import threading
from time import sleep, ctime



def direc2(a1_image,a2_image,n):      
    #threshold = 0.35
    sum0 = 0    
    # a1_image = cv2.cvtColor(a1_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("a1_image.jpg",a1_image)
    cv2.imwrite("a2_image.jpg",a2_image)
    h4,w4 = a2_image.shape[:2]
    ratio = w4/h4
    loops = []
    list_bool = []
    list_0 = []
    m= 7
    #m =  int(n*ratio)
    for i in range(n):#y
        for j in range(m):#x
            loops.append([i,j])
    threads = []
    nloops = range(len(loops))
    for i in nloops:
        t = MyThread(compare, (loops[i],a1_image,a2_image,n,m))#, compare.__name__))
        threads.append(t)
    for i in nloops: # start threads
        threads[i].start()
    for i in nloops: # wait for all
        threads[i].join() # threads to finish
    for i in nloops:
        value, max_val,res =  threads[i].get_result()
        # list_0.append(res)
        # sum0 += value
        list_bool.append(max_val)
        

    #print(f'all done at: {ctime()}')
    return sum0,list_bool,list_0

class MyThread(threading.Thread):
    def __init__(self, func, args):#, name=''):
        threading.Thread.__init__(self)
       # self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)
  
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def compare(coord,a1_image,a2_image,n,m):
    i = coord[0]
    j = coord[1]
    h , w = a1_image.shape[:2]
    threshold = 10000
   
    if i>0 and j>0 and i<n-1 and j<m-1:
        square1= a1_image[int((i-1)*h/n) : int((i+2)*h/n) , int((j-1)*w/m) : int((j+2)*w/m)]#裁剪为[y:y0,x:x0]
    elif i == 0 and j!=0 and j!=m-1:
        square1= a1_image[int((i)*h/n) : int((i+2)*h/n) , int((j-1)*w/m) : int((j+2)*w/m)]
    elif i == n-1 and j!=0 and j!=m-1:
        square1= a1_image[int((i-1)*h/n) : int((i+1)*h/n) , int((j-1)*w/m) : int((j+2)*w/m)]
    elif j == 0  and i!=0 and i!=n-1:
        square1= a1_image[int((i-1)*h/n) : int((i+2)*h/n) , int((j)*w/m) : int((j+2)*w/m)]
    elif j == m-1  and i!=0 and i!=n-1:
        square1= a1_image[int((i-1)*h/n) : int((i+2)*h/n) , int((j-1)*w/m) : int((j+1)*w/m)]
    elif j == m-1  and i==0 :
        square1= a1_image[int((i)*h/n) : int((i+2)*h/n) , int((j-1)*w/m) : int((j+1)*w/m)]
    elif j == m-1  and i== n-1 :
        square1= a1_image[int((i-1)*h/n) : int((i+1)*h/n) , int((j-1)*w/m) : int((j+1)*w/m)]
    elif j == 0 and i== n-1 :
        square1= a1_image[int((i-1)*h/n) : int((i+1)*h/n) , int((j)*w/m) : int((j+2)*w/m)]    
    elif j == 0 and i==0:
        square1= a1_image[int((i)*h/n) : int((i+2)*h/n) , int((j)*w/m) : int((j+2)*w/m)]      
    #square3= a1_image[int(i*h/n) : int((i+1)*h/n) , int(j*w/m) : int((j+1)*w/m)]
    
    square2= a2_image[int(i*h/n) : int((i+1)*h/n) , int(j*w/m) : int((j+1)*w/m)]
    
    res = cv2.matchTemplate(square1,square2,cv2.TM_CCOEFF_NORMED)#square1,原图
    #print(res)
    #res2 = cv2.matchTemplate(square3,square2,cv2.TM_CCOEFF)
    #list_1.append(res)
    #print(max(res[0][:]))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
    #print(max_val)
    return 0 ,max_val,res
    #print(res[0])
    # for i in range(len(res)):
    #     if max(res[i])== max_val:
    #         l = i
    #         break
    # #aver = sum(res[l])/len(res[l])
    # res_0 = [res[l],res2[0]]
  


    # if max_val<-10000:
    #     bool0 = 0
    # else:
    #     bool0 = 1
    # fo=open("threshold.txt", "a+")
    # fo.write(str(max_val)+"\t")
    #     # 关闭打开的文件
    # fo.close()
    # if max_val > threshold:
    #     return 1 ,bool0,res_0
    # else:

        # similar.append(1)



# if __name__ == "__main__":
#     path1="//home//wj//all//apparatus//"
#     # path1="//home//wj///all//imgs//img_match//high_shot//"
#     # path2 = "//home//wj//all//imgs//img_match//scan//"
#     n = 7
#     threshold = 0.35
    
#     for i in range(1):        
#         print("第"+str(i+1)+"张图,不同图片")
#         start = time.time()
#         a = cv2.imread(path1+"0.jpg")
#         a1 = cv2.imread(path1+"high_shot"+str(i+1)+".jpg")
#         a2 = cv2.imread(path1+"scan"+str(i+1)+".jpg")
#         #print("high_shot time:"+str(time.time()-start))
#         start = time.time() 
#         similar0,list_bool,list_3= direc2(a,a,10)
#         list8 = list(filter(lambda x: x ==1, list_bool))
#         #print(list8)
#         value0 = len(list8)
#         #print("value0"+str(value0))
#         #print("similar"+str(similar0))
#         #print("time"+str(time.time()-start))
#         if similar0 > int(n*n/2+6) and  value0 > n*n-10:
#             print(str(1)+"相同")
#         else:
#             print(str(0)+"不同")
#         #print("time"+str(time.time()))


