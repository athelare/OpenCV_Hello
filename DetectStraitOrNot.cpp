//
// Created by Jiyu_ on 2021/8/2.
//

#include "main.h"
#include "../cv_target_vision/src/utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

ObjectPointFinder* of = nullptr;

struct ROIElement{
    bool roiGot;
    bool drawing;
    Point pt1, pt2;
    Rect rect;
    int firstRowCol{};
    Mat firstRowTemplate;
    vector<Point> chosenPoints;
    vector<Point> lastPointPosition;
    vector<Mat> chosenPointsPat;

    ROIElement(): roiGot(false), drawing(false), pt1(0, 0),
                  pt2(0, 0), rect(), chosenPoints(), lastPointPosition(){}
    void clearROI(){
        //pt2 = pt1;
        roiGot = false;
    }
    void addChosenPoints(int x, int y){
        chosenPoints.emplace_back(x, y);
    }
    void popChosenPoints(){
        if(!chosenPoints.empty()){
            chosenPoints.pop_back();
            lastPointPosition.pop_back();
        }
    }
    bool validROI() const{
        return pt1.x != pt2.x && pt1.y != pt2.y;
    }
    Rect cornerPoints(){
        return rect = Rect(Point(max(min(pt1.x, pt2.x), 0), max(min(pt1.y, pt2.y), 0)),
                          Point(max(pt1.x, pt2.x), max(pt1.y, pt2.y)));
    }
    static int getMinDiffCol(const uchar* frameRow1, const uchar* frameRow2, int startCol1, int startCol2, int width, int range){
        int minDiff = 55555555, minDiffCol = 0;
        uchar p1, p2;
        for (int col = startCol2 - range; col < startCol2 + range; ++col){
            int curDiff = 0;
            for(int i=0;i<width;++i){
                p1 = frameRow1[startCol1 + i];
                p2 = frameRow2[col + i];
                int sqrtValue = (p1 > p2) ? (p1 - p2) : (p2 - p1);
                curDiff += sqrtValue * sqrtValue;
            }
            if (curDiff < minDiff) {
                minDiff = curDiff;
                minDiffCol = col;
            }
        }
        return minDiffCol;
    }
    vector<Point> getVerticalLine(const Mat& frameGray){
        uchar *frameRow1, *frameRow2, p1, p2;
        Rect selectedArea = cv::Rect(this->rect.tl(), this->rect.br());
        vector<Point>res;
        this->clearROI();
        //std::cout << selectedArea << std::endl;
        if(this->drawing) {
            firstRowCol = selectedArea.x;
            firstRowTemplate = frameGray(Rect(Point(0, selectedArea.y), Size(frameGray.cols, 1))).clone();
        }
        else{
            int minDiff = 55555555, minDiffCol = 0;
            frameRow1 = firstRowTemplate.data;
            frameRow2 = frameGray.data + frameGray.step * selectedArea.y;
            firstRowCol = getMinDiffCol(frameRow1, frameRow2, selectedArea.x, firstRowCol, selectedArea.width, selectedArea.width);
        }
        res.emplace_back(firstRowCol + this->rect.width/2, selectedArea.y);
        frameRow1 = frameGray.data + frameGray.step * selectedArea.y;
        int lastRowCol = firstRowCol;
        for(int row = selectedArea.y + 1; row < selectedArea.br().y; ++row){
            frameRow2 = frameGray.data + frameGray.step * row;
            lastRowCol = getMinDiffCol(frameRow1, frameRow2, firstRowCol, lastRowCol, selectedArea.width, 5);
            res.emplace_back(lastRowCol + this->rect.width/2, row);
        }
        //this->lastColPosition = vector<int>(this->initialColPosition.begin(), this->initialColPosition.end());
        return res;
    }

};

void mouseHandlerROI(int event, int x, int y, int flags, void* param){
    if(!param)return;
    auto* roi = (ROIElement*)param;
    if(event == cv::EVENT_LBUTTONDOWN){
        roi->roiGot = false;
        roi->drawing = true;
        roi->pt2 = roi->pt1 = cv::Point(x, y);
    }else if(roi->drawing && event == cv::EVENT_MOUSEMOVE){
        roi->pt2 = cv::Point(x, y);
        roi->cornerPoints();
    }else if(roi->drawing && event == cv::EVENT_LBUTTONUP){
        roi->drawing = false;
        roi->pt2 = cv::Point(x, y);
        if(roi->validROI()){
            roi->roiGot = true;
            roi->cornerPoints();
        }
    }else if(event == cv::EVENT_RBUTTONDOWN){
        roi->addChosenPoints(x, y);
    }else if(event == cv::EVENT_MBUTTONDOWN && of){
        of->findPoint(x, y);
        cout << "(" << of->retAddr()[0] << ", " << of->retAddr()[1] << ")" << endl;
    }
}

bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();

    //Construct matrix X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                                     std::pow(key_point[k].x, i + j);
            }
        }
    }

    //Construct matrix Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                                 std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //Solve matrix A
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return true;
}

Point2f getNearestCornerPosition(Mat&gray){
    Mat binary, bgrGray;
    vector<Point2f>corners;
    double minDistance = 100000.0;
    Point2f minPoint;
    TermCriteria tc = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

    //threshold(gray, binary, 100, 255, THRESH_OTSU);
    goodFeaturesToTrack(gray, corners, 5, 0.01, 4, Mat(), 3, false, 0.04);
    if(corners.empty()){
        return {(float)gray.cols/2, (float)gray.rows/2};
    }
    cornerSubPix(gray, corners, Size(5,5), Size(-1,-1), tc);

    cv::Point2f center = cv::Point(gray.cols/2, gray.rows/2);
    cvtColor(gray, bgrGray, COLOR_GRAY2BGR);
    for(const auto&corner: corners){
        circle(bgrGray, Point(corner.x, corner.y), 1, Scalar(0, 0, 255), 2);
        double dis = sqrt((corner.x-center.x)*(corner.x-center.x) + (corner.y-center.y)*(corner.y-center.y));
        if(dis < minDistance){
            minDistance = dis;
            minPoint = corner;
        }
    }

    imshow("asd1", bgrGray);
    //cout << minPoint << endl;
    return minPoint;
};

int DetectStraitOrNot(){
    const char ESC_KEY = 27;
    const char ENTER_KEY = 13;
    const char BACK_SPACE = 8;
    const Scalar COLOR_RED(0, 0, 255);
    const Scalar COLOR_GREEN(0, 255, 0);
    const Scalar COLOR_BLUE(255, 0, 0);


    // Camera
    //DVPFrameCapture cap;
    VideoCapture cap(1);
    cv::Mat frame, img, uFrame, frameGray;
    cv::Mat cameraMatrix, distCoeffs;
    int imageWidth, imageHeight;

    cv::FileStorage fs("out_camera_data_dvp_5MP_12mm.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["image_width"] >> imageWidth;
    fs["image_height"] >> imageHeight;
    fs.release();

    //Region of Interest
    ROIElement roi;
    cv::namedWindow("asd");
    cv::setMouseCallback("asd", mouseHandlerROI, &roi);


    double *objPosition;
    LocationParams locationParams;

    //cv::Mat templateFrame;
    cv::Rect selectedArea;
    bool verticalLineDetect = false;
    cv::Mat A;
    char tmpStr[64];

    while(true){
        cap.read(frame);

        cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        // 检测直线与检测点
        if(roi.validROI()) {
            vector<Point> linePoints = roi.getVerticalLine(frameGray);
            Point leftP(roi.firstRowCol, roi.rect.y), rightP(roi.firstRowCol + roi.rect.width, roi.rect.y);
            line(frame, leftP + Point(0, 5),  leftP + Point(0, -5), COLOR_RED, 2);
            line(frame, rightP + Point(0, 5),  rightP + Point(0, -5), COLOR_RED, 2);
            line(frame, leftP,  rightP, COLOR_RED, 2);
            polylines(frame, linePoints, false, COLOR_GREEN, 2);

            std::vector<cv::Point> newPointsReverse;
            newPointsReverse.reserve(linePoints.size());
            for (const auto& p:linePoints) {
                newPointsReverse.emplace_back(p.y, p.x);
            }
            polynomial_curve_fit(newPointsReverse, 1, A);
            // std::cout << "A = " << A << std::endl;
            sprintf(tmpStr, "X = %.4lf * Y + %.2lf", A.at<double>(1, 0), A.at<double>(0, 0));
            cv::putText(frame, tmpStr, linePoints[0], 1, 3, cv::Scalar(0, 0, 255), 3);
        }
        // 使用模板图匹配确定区域，距离区域中心最近的角点作为目标点
        int fieldHalfSize = 80;
        int patHalfSize = 25;
        Point patQuarterSquare(patHalfSize, patHalfSize);
        bool useCorner = false;
        for(int i = 0; i < roi.chosenPoints.size(); ++i) {
            if(roi.chosenPointsPat.size() <= i){
                if(useCorner){
                    Point offset(roi.chosenPoints[i] - patQuarterSquare);
                    Mat templatePart = frameGray(Rect(roi.chosenPoints[i] - patQuarterSquare, roi.chosenPoints[i] + patQuarterSquare));
                    Point centerPoint = getNearestCornerPosition(templatePart);
                    roi.lastPointPosition.emplace_back(offset + centerPoint);
                    roi.chosenPoints[i] = roi.lastPointPosition.back();
                    roi.chosenPointsPat.emplace_back(frameGray(Rect(roi.chosenPoints[i] - patQuarterSquare, roi.chosenPoints[i] + patQuarterSquare)).clone());

                }else{
                    roi.chosenPointsPat.emplace_back(frameGray(Rect(roi.chosenPoints[i] - patQuarterSquare, roi.chosenPoints[i] + patQuarterSquare)).clone());
                    roi.lastPointPosition.emplace_back(roi.chosenPoints[i]);
                }
                //cout << ROISelector::selectPoints[i] << " " << lastPointPosition[i] << endl;
            }
            Mat res;
            double minVal, maxVal;
            Point minLoc, maxLoc;
            Point offset2(roi.lastPointPosition[i].x - fieldHalfSize, roi.lastPointPosition[i].y - fieldHalfSize);
            Mat field = frameGray(cv::Rect(offset2, cv::Size(2 * fieldHalfSize, 2 * fieldHalfSize)));
            matchTemplate(field, roi.chosenPointsPat[i], res, cv::TM_CCOEFF);

            minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);
            rectangle(frame, offset2, offset2 + Point(fieldHalfSize * 2, fieldHalfSize * 2), COLOR_BLUE, 2);
            rectangle(frame, offset2 + maxLoc, offset2 + maxLoc + Point(patHalfSize * 2, patHalfSize * 2),
                          COLOR_RED, 2);
            if(useCorner){
                Mat matchPart = frameGray(Rect(offset2 + maxLoc, Size(2 * patHalfSize, 2 * patHalfSize)));
                roi.lastPointPosition[i] = (Point) getNearestCornerPosition(matchPart) + offset2 + maxLoc;
            } else {
                roi.lastPointPosition[i] = offset2 + maxLoc + patQuarterSquare;
            }

            //imshow("asd3", matchPart);


            cv::circle(frame, roi.lastPointPosition[i], 2, cv::Scalar(0, 255, 0), 2);

            if (!of) {
                sprintf(tmpStr, " (%d, %d)", roi.lastPointPosition[i].x, roi.lastPointPosition[i].y);
            } else {
                of->findPoint(roi.lastPointPosition[i].x, roi.lastPointPosition[i].y);
                sprintf(tmpStr, " (%.2lfmm, %.2lfmm)", objPosition[0], objPosition[1]);
            }


            cv::putText(frame, tmpStr, roi.lastPointPosition[i], 1, 3, cv::Scalar(0, 0, 255), 3);

        }


        //cv::resize(frame, img, cv::Size(1224, 1024));

        cv::imshow("asd", frame);

        int key = cv::waitKey(30);
        if(key == ESC_KEY)break;
        else if(key == BACK_SPACE){
            roi.popChosenPoints();
        }
        else if(key == ENTER_KEY){
            cap.read(frame);
            bool status = calcExParameters("out_camera_data_dvp_5MP_12mm.yml", frame, "calc_params.yml", 8, 5, 30);
            if(status){
                fs.open("calc_params.yml", FileStorage::READ);
                if(!fs.isOpened()){
                    cout << "Cannot open locationParams file." << endl;
                    return -1;
                }
                fs["location_params"] >> locationParams;
                fs.release();
                if(!locationParams.validate()){
                    cout << "Location parameters not provided completely. Exit." << endl;
                    return -1;
                }
                of = new ObjectPointFinder(cameraMatrix, distCoeffs, locationParams.getExMatrix(), locationParams.getB());
                objPosition = of->retAddr();
            }

            //            remap(frame, uFrame, map1, map2, cv::INTER_LINEAR);
//            cv::resize(uFrame, img, cv::Size(1224, 1024));
//            cv::imshow("asd", img);
//            cv::waitKey();
//
//            cv::cvtColor(uFrame, gray, cv::COLOR_BGR2GRAY);
//            cv::Canny(gray, gray, 50, 150, 3);
//            cv::resize(gray, img, cv::Size(1224, 1024));
//            cv::imshow("asd", img);
//            cv::waitKey();
//
////            std::vector<cv::Vec2f> lines;
////            cv::HoughLines(gray, lines, 1, CV_PI/180, 400, 0, 0);
////            for (auto & line : lines) {
////                float rho = line[0];
////                float theta = line[1];
////                cv::Point pt1, pt2;
////                double a = cos(theta), b = sin(theta);
////                double x0 = a * rho, y0 = b * rho;
////                pt1.x = cvRound(x0 + 1000 * (-b));
////                pt1.y = cvRound(y0 + 1000 * (a));
////                pt2.x = cvRound(x0 - 1000 * (-b));
////                pt2.y = cvRound(y0 - 1000 * (a));
////                cv::line(uFrame, pt1, pt2, cv::Scalar(0, 0, 255),2);
////            }
//            std::vector<cv::Vec4i> lines;
//            cv::HoughLinesP(gray, lines, 1, CV_PI/180, 200, 50);
//            for(auto&line:lines){
//                cv::line(uFrame, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255),2);
//            }
//
//            cv::resize(uFrame, img, cv::Size(1224, 1024));
//            cv::imshow("asd", img);
//            cv::waitKey();

        }
    }
    return 0;
}