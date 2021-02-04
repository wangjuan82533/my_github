#include "opencv2/imgproc.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/traits.hpp"
#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
string high_shot(Mat,string);
string line_detect_possible_demo(Mat, vector<Vec4i>&,int & ,int&);
Point cross_point(Vec4i, Vec4i);
Point2f Euclidean(vector<Point>, int, int);
void deleteneg(vector<Point>&);

int main(){
    string path = "/home/wj/all/imgs/apparatus/38.jpg";
    string outpath = "./result.jpg";
    Mat img = imread(path);
    string result=high_shot(img,outpath);
    cout << result << endl;
    return 0;
}

string high_shot(Mat img,string outpath){
    vector<Vec4i> lines;
    int xx, yy;
    int count = 0;
    string result = line_detect_possible_demo(img, lines, xx, yy);
    if (result != "ok")
    {
        return result;
    }
    else{
        vector<vector<double>> list_k;
        vector<vector<double>> list_k2;
        vector<string> list_k1;
        vector<Point> list_point,list_point1;
        for (int i = 0; i != lines.size(); i++)
        {
            for (int j = 0; j != lines.size();j++){
                Point point1(lines[i][0], lines[i][1]);
                Point point2(lines[i][2], lines[i][3]) ;
                Point point3(lines[j][0], lines[j][1]);
                Point point4(lines[j][2], lines[j][3]);
                if (lines[i][2] != lines[i][0] && lines[j][2] != lines[j][0])
                {
                    double k1 = (lines[i][3] - lines[i][1]) / (float(lines[i][2] - lines[i][0]));
                    double k2 = (lines[j][3] - lines[j][1]) / (float(lines[j][2] - lines[j][0]));
                    double cobb;
                    if (k1 * k2 == -1)
                    {
                        cobb = 90.0;
                    }else{
                        cobb = fabs(atan((k1 - k2) / (double(1 + k1 * k2))) * 180 / 3.1415926 )+ 0.5;
                    }
                    if (cobb < 95 && cobb > 85)
                    {

                        list_k1.push_back("ok");
                        Point cross = cross_point(lines[i], lines[j]);
                        count++;
                        list_point.push_back(cross);
                        vector<double> list_temp;
                        list_temp.push_back(k1);
                        list_temp.push_back(k2);
                        list_k.push_back(list_temp);
                    }
                    else if (cobb < 5 && fabs(k1) < 0.0875 && fabs(k2) < 0.0875)
                    {
                        string k1 = "double0";
                        list_point1.push_back(point1);
                        list_point1.push_back(point2);
                        list_point1.push_back(point3);
                        list_point1.push_back(point3);
                        list_k1.push_back(k1);
                    }
                    else
                    {
                        string k1 = "未找到纸张";
                        list_k1.push_back(k1);
                    }
                    
                }
                else if (lines[i][2] != lines[i][0] && lines[j][2] == lines[j][0])
                {
                    float k1 = (lines[i][3] - lines[i][1]) / (float(lines[i][2] - lines[i][0]));
                    if(fabs(atan(fabs(k1))*180/3.1415926)<=5){
                        list_point.push_back(cross_point(lines[i], lines[j]));
                        list_k1.push_back(" single无穷大");
                    }
                }
                else if (lines[i][2] == lines[i][0] && lines[j][2] != lines[j][0])
                {
                    float k1 = (lines[j][3] - lines[j][1]) / (float(lines[j][2] - lines[j][0]));
                    if(fabs(atan(fabs(k1))*180/3.1415926)<=5){
                        list_point.push_back(cross_point(lines[i], lines[j]));
                        list_k1.push_back(" single无穷大");
                    }
                }
                else
                {
                    string k1 = "无穷大";
                    // list_point.push_back(point1);
                    // list_point.push_back(point2);
                    // list_point.push_back(point3);
                    // list_point.push_back(point4);
                    list_k1.push_back(k1);
                }
            }
        }
        int num1 = 0;
        int num2 = 0;
        for (int i = 0; i != list_k1.size(); i++)
        {
            if(list_k1[i]!="未找到纸张"){
                num1 = 1;
            }
            if(list_k1[i]!="double0"){
                num2 = 1;
            }
        }
            vector<Point> vec_0, vec_1, vec_2, vec_3;
        if (num1 == 0)
        {
            return "未找到纸张";
        }
        else if(num2==0 && list_point.size()==0){
            for (int i = 0; i != list_point1.size(); i++)
                {
                if(list_point1[i].y>yy &&list_point1[i].x<xx){
                    vec_0.push_back(list_point1[i]);
                }
                else if (list_point1[i].y < yy && list_point1[i].x < xx)
                {
                    vec_1.push_back(list_point1[i]);
                }
                else if (list_point1[i].y > yy && list_point1[i].x > xx)
                {
                    vec_2.push_back(list_point1[i]);
                }
                else if (list_point1[i].y < yy && list_point1[i].x > xx)
                {
                    vec_3.push_back(list_point1[i]);
                }
            }    
        }
        else if(list_k.size()!=0)
        {
            deleteneg(list_point);
            double kx, ky;
            double k1 = list_k[0][0];
            double k2 = list_k[0][1];
            if (k1 >  0 && k2 < 0)
            {
                kx = k1;
                ky = k2;
                for (int i = 0; i != list_point.size();i++){
                    if((list_point[i].y -(ky*list_point[i].x+yy-ky*xx))>0 && (list_point[i].y-(kx*list_point[i].x+yy-kx*xx))>0){
                        vec_0.push_back(list_point[i]);
                    }
                    else if ((list_point[i].y - (ky * list_point[i].x + yy - ky * xx)) < 0 && (list_point[i].y - (kx * list_point[i].x + yy - kx * xx)) > 0)
                    {
                        vec_1.push_back(list_point[i]);
                    }
                    else if ((list_point[i].y - (ky * list_point[i].x + yy - ky * xx)) >0 && (list_point[i].y - (kx * list_point[i].x + yy - kx * xx)) < 0)
                    {
                        vec_2.push_back(list_point[i]);
                    }
                    else if ((list_point[i].y - (ky * list_point[i].x + yy - ky * xx)) <0 && (list_point[i].y - (kx * list_point[i].x + yy - kx * xx)) < 0)
                    {
                        vec_3.push_back(list_point[i]);
                    }
                }
            }
            else if (k1 < 0 && k2 > 0)
            {
                kx = k2;
                ky = k1;
                for (int i = 0; i != list_point.size();i++){
                    if((list_point[i].y -(ky*list_point[i].x+yy-ky*xx))>0 && (list_point[i].y-(kx*list_point[i].x+yy-kx*xx))>0){
                        vec_0.push_back(list_point[i]);
                    }
                    else if ((list_point[i].y - (ky * list_point[i].x + yy - ky * xx)) < 0 && (list_point[i].y - (kx * list_point[i].x + yy - kx * xx)) > 0)
                    {
                        vec_1.push_back(list_point[i]);
                    }
                    else if ((list_point[i].y - (ky * list_point[i].x + yy - ky * xx)) >0 && (list_point[i].y - (kx * list_point[i].x + yy - kx * xx)) < 0)
                    {
                        vec_2.push_back(list_point[i]);
                    }
                    else if ((list_point[i].y - (ky * list_point[i].x + yy - ky * xx)) <0 && (list_point[i].y - (kx * list_point[i].x + yy - kx * xx)) < 0)
                    {
                        vec_3.push_back(list_point[i]);
                    }
                }
            }
            else{
                for (int i = 0; i != list_point.size(); i++)
                {
                    if(list_point[i].y>yy &&list_point[i].x<xx){
                        vec_0.push_back(list_point[i]);
                    }
                    else if (list_point[i].y < yy && list_point[i].x < xx)
                    {
                        vec_1.push_back(list_point[i]);
                    }
                    else if (list_point[i].y > yy && list_point[i].x > xx)
                    {
                        vec_2.push_back(list_point[i]);
                    }
                    else if (list_point[i].y < yy && list_point[i].x > xx)
                    {
                        vec_3.push_back(list_point[i]);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i != list_point.size(); i++)
            {
                if(list_point[i].y>yy &&list_point[i].x<xx){
                    vec_0.push_back(list_point[i]);
                }
                else if (list_point[i].y < yy && list_point[i].x < xx)
                {
                    vec_1.push_back(list_point[i]);
                }
                else if (list_point[i].y > yy && list_point[i].x > xx)
                {
                    vec_2.push_back(list_point[i]);
                }
                else if (list_point[i].y < yy && list_point[i].x > xx)
                {
                    vec_3.push_back(list_point[i]);
                }
            }
        }
        if (vec_0.size() == 0 || vec_1.size() == 0 || vec_2.size() == 0 || vec_3.size() == 0)
        {
            return "文件不平整,破损或未放入";
        }
        else{
            Point rightdown = Euclidean(vec_2, xx, yy);
            Point leftdown = Euclidean(vec_0, xx, yy);
            Point rightup = Euclidean(vec_3, xx, yy);
            Point leftup = Euclidean(vec_1, xx, yy);
            int h = int(sqrt((leftup.x - leftdown.x) * (leftup.x - leftdown.x) + (leftup.y - leftdown.y) * (leftup.y - leftdown.y)));
            int w = int(sqrt((leftup.x - rightup.x) * (leftup.x - rightup.x) + (leftup.y - rightup.y) * (leftup.y - rightup.y)));
            if(h<=w){
                rightdown = Euclidean(vec_0, xx, yy);
                leftdown = Euclidean(vec_1, xx, yy);
                rightup = Euclidean(vec_2, xx, yy);
                leftup = Euclidean(vec_3, xx, yy); 
                h = int(sqrt((leftup.x - leftdown.x) * (leftup.x - leftdown.x) + (leftup.y - leftdown.y) * (leftup.y - leftdown.y)));
                w = int(sqrt((leftup.x - rightup.x) * (leftup.x - rightup.x) + (leftup.y - rightup.y) * (leftup.y - rightup.y)));
            }
            vector< Point2f> roi_corners;
            roi_corners.push_back(leftdown);
            roi_corners.push_back(rightdown);
            roi_corners.push_back(rightup);
            roi_corners.push_back(leftup);
            vector<Point2f> dst_corners(4);
            dst_corners[0].x = 0;
            dst_corners[0].y = h;
            dst_corners[1].x = w;
            dst_corners[1].y = h;
            dst_corners[2].x = w;
            dst_corners[2].y = 0;
            dst_corners[3].x = 0;
            dst_corners[3].y = 0;
            Mat M = getPerspectiveTransform(roi_corners, dst_corners);
            Mat warped_image;
            warpPerspective(img, warped_image, M,Size(w,h));
            // warped_image(Range(0,h), Range(0, int(w / 256))) = 255;
            // warped_image(Range(0, h), Range(int(w*255/256), w)) = 255;
            // warped_image(Range(0, int(h/255)), Range(0, w)) = 255;
            // warped_image(Range(int(h*255/256), h), Range(0, w)) = 255;
            imwrite(outpath, warped_image);
            return "ok";
        }
    }
}


void deleteneg(vector<Point> &vec){
    int i = 0;
    while(i<vec.size()){
        if (vec[i].x <= -1)
        {
            vec.erase(vec.begin() + i);
            }else{
                ++i;
            }
    }
}



Point2f Euclidean(vector<Point> vec,int xx,int yy){
    map<int, Point> mymap;
    vector<int> vec_a;
    for (int i = 0; i != vec.size(); i++)
    {
        int a = pow((vec[i].x - xx),2)  + pow((vec[i].y - yy),2);
        mymap[a] = vec[i];
        vec_a.push_back(a);
        if(a<0){
        }
    }
    int min_a = *min_element(vec_a.begin(), vec_a.end());
    Point2f point_0 = mymap.find(min_a)->second;
    return point_0;
}

string line_detect_possible_demo(Mat img_target, vector<Vec4i> &lines,int &xx,int &yy)
{
    Mat gray, blur,thresh;
    int m;
    cvtColor(img_target, gray, COLOR_BGR2GRAY); //总结，转灰度
    GaussianBlur(gray, blur, Size(15, 15), 0);//总结，高斯滤波
    threshold(blur, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);//总结，二值化
    int h_thre = thresh.rows;
    int w_thre = thresh.cols;
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (int i = 0; i != contours.size();i++){
        for (int j = 0; j != contours[i].size();j++){
            if(contours[i][j].y  < int(h_thre*4/5)){
                m = i;
                break;
            }else{
                m = -1;
            }
        }
    }
    if(m==-1){
        return "未找到文件";
    }
    else
    {
        int sum0 =  0,sum1 =0, x_max = 0;
        int x_min = w_thre;
        int y_max = contours[m][0].y;
        int y_min = contours[m][0].y;
        for (int i = 0; i != contours[m].size();i++){
            sum0 += contours[m][i].y;
            sum1 += contours[m][i].x;
            if(x_max<contours[m][i].x){
                x_max = contours[m][i].x;
            }
            if(x_min>contours[m][i].x){
                x_min = contours[m][i].x;
            }
            if(y_max<contours[m][i].y){
                x_max = contours[m][i].y;
            }
            if(x_min>contours[m][i].y){
                x_min = contours[m][i].y;
            }
        }
        yy = int(sum0 / (contours[m].size()));
        xx = int(sum1 / (contours[m].size()));
        int x_dis = x_max - x_min;
        if(y_max>h_thre*31/32 || x_min<w_thre/32 ||x_max>w_thre*31/32 ||y_min<h_thre/32){
            return "位置异常，请重新放置";
        }
        else{
            if(x_dis>w_thre*31/32){
                x_max = 0;
                x_min = w_thre;
                for (int i = 0; i != contours[m].size();i++){
                    if(contours[m][i].y < yy){
                        if(x_max<contours[m][i].x){
                            x_max = contours[m][i].x;
                        }
                        if(contours[m][i].x<x_min){
                            x_min = contours[m][i].x;
                        }
                    }
                }
            }
            int x_dis = x_max - x_min;
            int y_dis = y_max - y_min;
            int h_temp;
            if (x_dis >= y_dis)
            {
                h_temp = x_dis / 2;
            }else{
                h_temp = y_dis / 2;
            }
            Mat gray0,res;
            
            Mat drawSrc = Mat(Size(w_thre, h_thre), CV_8UC3, Scalar(0, 0, 0));
            drawContours(drawSrc, contours, m, Scalar(255, 255, 255), 5);
            cvtColor(drawSrc, gray0, COLOR_BGR2GRAY);
            threshold(gray0, res, 0, 255, THRESH_BINARY + THRESH_OTSU);
            imwrite("./res.jpg", res);
            HoughLinesP(res, lines, 1, 3.1415926 / 180, 300, h_temp / 5, 100);
            if(lines.size()==0){
                return "未找到纸张";
            }
            else{
                if(lines.size()>45){
                     HoughLinesP(res, lines, 1, 3.1415926 / 180, 400, h_temp / 3.5, 300);
                }
                cout << "拟合直线数" << lines.size()<<endl;
                return "ok";
            }
        }
    }
}



Point cross_point(Vec4i vec_0, Vec4i vec_1){
    Point point;
    float x1 = vec_0[0];
    float x2 = vec_0[2];
    float y1 = vec_0[1];
    float y2 = vec_0[3];

    float x3 = vec_1[0];
    float x4 = vec_1[2];
    float y3 = vec_1[1];
    float y4 = vec_1[3];
    float x, y;
    float k1, k2, b1, b2;
    if (x1 != x2)
    {
        k1 = (y2 - y1) / (x2 - x1);
        b1 = y1- x1 * k1 ;
        if(x4==x3){
            x = x3;
            y = k1 * x + b1;
        }else{
            k2 = (y4 - y3) / (x4 - x3);
            b2 = y3 - x3 * k2 ;
            if(k1!=k2){
                x = (b2 - b1)  / (k1 - k2);
                y = k1 * x + b1 ;
            }else{
                x = -1;
                y = -1;
            }
        }
    }
    else{
        if(x4-x3==0){
            x = -1;
            y = -1;
        }else{
            k2 = (y4 - y3)  / (x4 - x3);
            b2 = y3  - x3 * k2 ;
            x = x1;
            y = k2 * x + b2 ;
        }
    }
    point.x = int(x);
    point.y = int(y);

    return point;
}