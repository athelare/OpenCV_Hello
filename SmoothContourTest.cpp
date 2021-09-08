//
// Created by Jiyu_ on 2021/8/30.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "main.h"
using namespace std;
using namespace cv;

void makeContourSmoother5(const vector<Point>&ctr, vector<Point>& outputArray, int k){
    Mat smoothCont(ctr), result;
    copyMakeBorder(smoothCont, smoothCont, (k-1)/2,(k-1)/2 ,0, 0, cv::BORDER_WRAP);
    blur(smoothCont, result, cv::Size(1,k),cv::Point(-1,-1));
    result.rowRange(Range((k-1)/2,1+result.rows-(k-1)/2)).copyTo(outputArray);
}

int SmoothContourTest(){
    Mat img;
    Scalar COLOR_BLUE = Scalar(255, 0, 0);
    Scalar COLOR_GREEN = Scalar(0, 255, 0);
    Scalar COLOR_RED = Scalar(0, 0, 255);
    img = imread("20210830_113425_contour_circle_Y.jpg", IMREAD_GRAYSCALE);
    threshold(img, img, 140, 255, THRESH_OTSU);
    imshow("win1", img);

    vector<vector<Point>> contours;
    findContours(img, contours, noArray(), RETR_CCOMP, CHAIN_APPROX_NONE);

    cout << contours.size() << endl;

    int max_area_index = 0;
    double max_area = -1, current_area;
    for (int i = 0; i < contours.size(); ++i) {
        current_area = contourArea(contours[i]);
        if (current_area > max_area) {
            max_area_index = i;
            max_area = current_area;
        }
    }

    vector<vector<Point>> v(1);
    Mat smoothCont = Mat(contours[max_area_index]);
    smoothCont.copyTo(v[0]);
    Mat img1;
    cvtColor(img, img1, COLOR_GRAY2BGR);
    drawContours(img1, v, 0, COLOR_RED);
    imshow("win2", img1);

    int k = 20;
    Mat result;
    makeContourSmoother5(contours[max_area_index], v[0], 20);
    Mat img2;
    cvtColor(img, img2, COLOR_GRAY2BGR);
    drawContours(img2, v, 0, COLOR_RED);
    imshow("win3", img2);
    imwrite("contour_2.png", img2);


    waitKey();



    return 0;
}