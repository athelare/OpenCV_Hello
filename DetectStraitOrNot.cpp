//
// Created by Jiyu_ on 2021/8/2.
//

#include "DvpFrameCapture.h"
#include "utils.h"
#include <iostream>

ObjectPointFinder* of = nullptr;

struct ROIElement{
    bool roiGot;
    bool drawing;
    Point pt1, pt2;
    Rect rect;
    vector<Point> chosenPoints;
    vector<Point> lastPosition;

    ROIElement():roiGot(false), drawing(false), pt1(0, 0),
        pt2(0, 0), rect(), chosenPoints(), lastPosition(){}
    void clearROI(){
        pt2 = pt1 = Point(0, 0);
        drawing = roiGot = false;
    }
    void addChosenPoints(int x, int y){
        chosenPoints.emplace_back(x, y);
    }
    void popChosenPoints(){
        if(!chosenPoints.empty()){
            chosenPoints.pop_back();
            lastPosition.pop_back();
        }
    }
    bool validROI() const{
        return pt1.x != pt2.x && pt1.y != pt2.y;
    }
    Rect cornerPoints(){
        return rect = Rect(Point(max(min(pt1.x, pt2.x), 0), max(min(pt1.y, pt2.y), 0)),
                          Point(max(pt1.x, pt2.x), max(pt1.y, pt2.y)));
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
    }else if(roi->drawing && event == cv::EVENT_LBUTTONUP){
        roi->drawing = false;
        roi->pt2 = cv::Point(x, y);
        if(roi->validROI()){
            roi->roiGot = true;
            roi->cornerPoints();
        }
    }else if(event == cv::EVENT_RBUTTONDOWN){
        roi->addChosenPoints(x*2, y*2);
    }else if(event == cv::EVENT_MBUTTONDOWN && of){
        of->findPoint(x*2, y*2);
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


int main(){
    dvpCameraInfo info[8];
    int cameraCount = 0;
    const char ESC_KEY = 27;
    const char ENTER_KEY = 13;
    const char BACK_SPACE = 8;
    dvpUint32 count = 0;
    /* 枚举设备 */
    dvpRefresh(&count);
    if (count > 8)
        count = 8;
    for (int i = 0; i < 8; i++)
    {
        if (dvpEnum(i, &info[i]) == DVP_STATUS_OK)
        {
            printf("[%d]-Camera FriendlyName : %s\r\n", i, info[i].FriendlyName);
            cameraCount++;
        }
    }
    if(cameraCount <= 0){
        std::cout << "No DVP device found.";
        return -1;
    }
    // Camera
    DVPFrameCapture cap((char*)info[0].FriendlyName);
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

    cv::Mat templateFrame;
    cv::Rect selectedArea;
    vector<int>lastColPosition;
    bool startDetect = false;
    cv::Mat A;
    char tmpStr[64];

    while(true){
        cap.nextFrame(frame);
        if(templateFrame.empty()){
            cv::cvtColor(frame, templateFrame, cv::COLOR_BGR2GRAY);
        }


        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        // 检测直线与检测点
        if(startDetect) {
            vector<Point> newPoints;
//            if(lastColPosition.empty()){
//                lastColPosition = vector<int>(selectedArea.height, selectedArea.x);
//            }
            for (int row = selectedArea.y; row < selectedArea.br().y; ++row) {
                int minDiff = 55555555;
                int minDiffCol = 0;
                uchar* frameRow1;
                uchar* frameRow2;
                frameRow1 = frameGray.data + row * frameGray.step;
                frameRow2 = templateFrame.data + row * templateFrame.step;

                for (int col = lastColPosition[row - selectedArea.y] - 2*selectedArea.width; col < lastColPosition[row - selectedArea.y] + 2*selectedArea.width; ++col) {
                    int curDiff = 0;
                    for (int i = 0; i < selectedArea.width; ++i) {
                        uchar p1, p2;
                        p1 = frameRow2[selectedArea.x + i];
                        p2 = frameRow1[col + i];
                        int sqrtValue = (p1 > p2) ? (p1 - p2) : (p2 - p1);
                        curDiff += sqrtValue * sqrtValue;
                    }
                    if (curDiff < minDiff) {
                        minDiff = curDiff;
                        minDiffCol = col;
                    }
                }
                lastColPosition[row - selectedArea.y] = minDiffCol;
                newPoints.emplace_back(minDiffCol, row);
            }

            cv::polylines(frame, newPoints, false, cv::Scalar(0, 255, 0), 2);

            std::vector<cv::Point> newPointsReverse;
            newPointsReverse.reserve(newPoints.size());
            for (const auto& p:newPoints) {
                newPointsReverse.emplace_back(p.y, p.x);
            }
            polynomial_curve_fit(newPointsReverse, 1, A);
            // std::cout << "A = " << A << std::endl;
            sprintf(tmpStr, "X = %.4lf * Y + %.2lf", A.at<double>(1, 0), A.at<double>(0, 0));
            cv::putText(frame, tmpStr, newPoints[0], 1, 3, cv::Scalar(0, 0, 255), 3);
        }
        // 使用模板图匹配确定区域，距离区域中心最近的角点作为目标点
        int matchSize = 80;
        int patSize = 25;
        for(int i = 0; i < roi.chosenPoints.size(); ++i) {
            if(roi.lastPosition.size() <= i){
                Point offset(roi.chosenPoints[i].x - patSize, roi.chosenPoints[i].y - patSize);
                Mat templatePart = templateFrame(Rect(offset, Size(2 * patSize, 2 * patSize)));
                roi.lastPosition.emplace_back(offset + (Point)getNearestCornerPosition(templatePart));
                roi.chosenPoints[i] = roi.lastPosition.back();
                //cout << ROISelector::selectPoints[i] << " " << lastPointPosition[i] << endl;
            }
            cv::Mat res;
            cv::Point offset1(roi.chosenPoints[i].x - patSize, roi.chosenPoints[i].y - patSize);
            cv::Point offset2(roi.lastPosition[i].x - matchSize, roi.lastPosition[i].y - matchSize);
            cv::Mat pat = templateFrame(cv::Rect(offset1, cv::Size(2 * patSize, 2 * patSize)));
            cv::Mat field = frameGray(cv::Rect(offset2, cv::Size(2 * matchSize, 2 * matchSize)));
            cv::matchTemplate(field, pat, res, cv::TM_CCOEFF);
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);
            cv::rectangle(frame, offset2 + maxLoc, offset2 + maxLoc + cv::Point(patSize * 2, patSize * 2),
                          cv::Scalar(0, 0, 255), 2);


            Mat matchPart = frameGray(Rect(offset2 + maxLoc, Size(2 * patSize, 2 * patSize)));
//            Mat binary, dst;
//            vector<Point2f>corners;
//
//            threshold(matchPart, binary, 100, 255, THRESH_OTSU);
//            //cornerHarris(matchPart, dst, 2, 3, 0.04);
//            goodFeaturesToTrack(matchPart, corners, 5, 0.01, 10, Mat(), 3, false, 0.04);
//            TermCriteria tc = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);
//            cornerSubPix(matchPart, corners, Size(5,5), Size(-1,-1), tc);
//            //minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);
//            //threshold(dst, dst, 0.01 * maxVal, 255, THRESH_BINARY);
//            double minDistance = 100000.0;
//            Point2f minPoint;
//            cv::Point center = cv::Point(patSize, patSize);
//            for(const auto&corner: corners){
//                circle(matchPart, Point(corner.x, corner.y), 1, 255, 2);
//                double dis = sqrt((corner.x-center.x)*(corner.x-center.x) + (corner.y-center.y)*(corner.y-center.y));
//                if(dis < minDistance){
//                    minDistance = dis;
//                    minPoint = corner;
//                }
//            }
//            imshow("asd1", matchPart);



            roi.lastPosition[i] = (Point) getNearestCornerPosition(matchPart) + offset2 + maxLoc;
            cv::circle(frame, roi.lastPosition[i], 2, cv::Scalar(0, 255, 0), 2);

            if (!of) {
                sprintf(tmpStr, " (%d, %d)", roi.lastPosition[i].x, roi.lastPosition[i].y);
            } else {
                of->findPoint(roi.lastPosition[i].x, roi.lastPosition[i].y);
                sprintf(tmpStr, " (%.2lfmm, %.2lfmm)", objPosition[0], objPosition[1]);
            }


            cv::putText(frame, tmpStr, roi.lastPosition[i], 1, 3, cv::Scalar(0, 0, 255), 3);

        }



//            int range = 25;
//            int patSize = 25;
//            for(const cv::Point& selectPoint:ROISelector::selectPoints){
//                int minX=0, minY=0, minDiff = 55555555;
//                for(int y = selectPoint.y-range;y<selectPoint.y+range;++y){
//                    for(int x = selectPoint.x - range; x<selectPoint.x + range;++x){
//                        int curDiff = 0;
//                        for(int i=-patSize; i<patSize;++i){
//                            for(int j = -patSize;j<patSize;++j){
//                                uchar p1, p2;
//                                p1 = oldFrame.at<uchar>(selectPoint.y + i, selectPoint.x + j);
//                                p2 = frame.at<uchar>(y + i, x + j);
//                                curDiff += (p1>p2)?(p1-p2):(p2-p1);
//                            }
//                        }
//                        if(curDiff < minDiff){
//                            minDiff = curDiff;
//                            minX = x;
//                            minY = y;
//                        }
//                    }
//                }
//                sprintf(tmpStr, "(%d, %d)", minX, minY);
//                cv::putText(frame, tmpStr, cv::Point(minX, minY), 1, 3, cv::Scalar(0, 0, 255), 3);
//                cv::circle(frame, cv::Point(minX, minY), 2, cv::Scalar(0, 0, 255), 2);

        cv::resize(frame, img, cv::Size(1224, 1024));
        if(roi.validROI()){
            startDetect = false;
            roi.cornerPoints();
            cv::rectangle(img, roi.rect, cv::Scalar(0, 0, 255), 2);
        }

        if(roi.roiGot){
            selectedArea = cv::Rect(roi.rect.tl()*2, roi.rect.br()*2);
            roi.clearROI();
            cv::cvtColor(frame, templateFrame, cv::COLOR_BGR2GRAY);
            startDetect = true;
            //std::cout << selectedArea << std::endl;
            lastColPosition.clear();
            lastColPosition.emplace_back(selectedArea.x);
            for(int row = selectedArea.y + 1; row < selectedArea.br().y; ++row){
                int minDiff = 55555555;
                int minDiffCol = 0;
                uchar *frameRow1, *frameRow2, p1, p2;
                frameRow1 = templateFrame.data + templateFrame.step * selectedArea.y;
                frameRow2 = templateFrame.data + templateFrame.step * row;
                int lastRowCol = lastColPosition[row - selectedArea.y - 1];
                for (int col = lastRowCol - 2 * selectedArea.width; col < lastRowCol + 2 * selectedArea.width; ++col){
                    int curDiff = 0;
                    for(int i=0;i<selectedArea.width;++i){
                        p1 = frameRow1[selectedArea.x + i];
                        p2 = frameRow2[col + i];
                        int sqrtValue = (p1 > p2) ? (p1 - p2) : (p2 - p1);
                        curDiff += sqrtValue * sqrtValue;
                    }
                    if (curDiff < minDiff) {
                        minDiff = curDiff;
                        minDiffCol = col;
                    }
                    //cout << curDiff << endl;
                }
                lastColPosition.emplace_back(minDiffCol);
            }
            cout << "ROI got." << endl;
        }



        cv::imshow("asd", img);

        int key = cv::waitKey(30);
        if(key == ESC_KEY)break;
        else if(key == BACK_SPACE){
            roi.popChosenPoints();
        }
        else if(key == ENTER_KEY){
            cap.nextFrame(frame);
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