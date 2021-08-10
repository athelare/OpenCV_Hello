#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;





int mai3n(){
    Mat img = cv::imread("hole.png", 0);
    Mat colorImg;
    cvtColor(img, colorImg, cv::COLOR_GRAY2BGR);

    vector<Vec3f> circles;
    HoughCircles(img, circles, HOUGH_GRADIENT,
                     2, 100, 100, 15, 20, 25);

    for(auto & i : circles)
    {
        cout << i << endl;
        Point center(cvRound(i[0]), cvRound(i[1]));
        int radius = cvRound(i[2]);
        // draw the circle center
        circle( colorImg, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // draw the circle outline
        circle( colorImg, center, radius, Scalar(0,0,255), 1, 8, 0 );
    }

    imshow("circles", colorImg);
    waitKey();
    return 0;
}