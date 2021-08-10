#include<iostream>
#include<opencv2/opencv.hpp>

#include "include/DVPCamera.h"

using namespace std;
using namespace cv;

int main0(){

    VideoCapture cap(0);
    cout << cap.isOpened() << endl;
    while(cap.isOpened()){
        Mat frame;
        cap >> frame;
        imshow("Camera view", frame);
        int key = waitKey(30);
        if(key == 27)break;
    }

    return 0;
}