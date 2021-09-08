//
// Created by Jiyu_ on 2021/9/4.
//

#include "main.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int cannyTest(){
    Mat holeImg, dst1, dst2, dst3, imgBlur;
    holeImg = imread("circleCenter.png", 0);
    GaussianBlur(holeImg, imgBlur, Size(7, 7), 0);
    Canny(imgBlur, dst1, 40, 20);
    imshow("circle", holeImg);
    imshow("blur", imgBlur);
    imshow("asd", dst1);

    vector<Vec3f> circles;
    HoughCircles(holeImg, circles, HOUGH_GRADIENT,
                 3, 1, 60, 15,
                 42, 46);

    Mat colorHoleImg;
    cvtColor(holeImg, colorHoleImg, COLOR_GRAY2BGR);
    cout << " " <<  circles.size() << " circles detected." << endl;

    for(int i=0;i<3 && i<circles.size();++i){
        cout << "(" << circles[i][0] << ", " << circles[i][1] << ") " << circles[i][2] << endl;
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        circle(colorHoleImg, center, 3 - i, Scalar(255, 0, 0), -1, 8, 0);
        circle(colorHoleImg, center, cvRound(circles[i][2]), Scalar(0, 0, 255), 1, 8, 0);
    }

    //resize(colorHoleImg, colorHoleImg, Size(2 * colorHoleImg.cols, 2 * colorHoleImg.rows));
    imshow("hole", colorHoleImg);
    waitKey();

    return 0;
}

