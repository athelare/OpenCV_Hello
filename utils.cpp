#include<iostream>
#include<queue>
#include<stack>
#include "utils.h"

using namespace std;
using namespace cv;

static void calcChessboardCornerPositions(const Size& boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.clear();
    for( int i = 0; i < boardSize.height; ++i )
        for( int j = 0; j < boardSize.width; ++j )
            corners.emplace_back(Point3f(j*squareSize, i*squareSize, 0));
}

void dot(const Mat&A, const Mat&B, Mat&C){
    if(A.cols != B.rows)return;
    C=Mat (A.rows, B.cols, CV_64F);
    for(int i=0;i<A.rows;++i){
        for(int j=0;j<B.cols;++j){
            double cij = 0.;
            for(int k=0;k<A.cols;++k){
                cij += A.at<double>(i, k)*B.at<double>(k, j);
            }
            C.at<double>(i, j) = cij;
        }
    }
}

vector<Point> expandCorners(vector<Point2f>&corners, const Size& patternSize, int expandLen){
    vector<Point2f>contour;
    vector<Point>contour_ex;
    int w = patternSize.width, h = patternSize.height;
//    vector<vector<Point2f>> chessboardCorners(h, vector<Point2f>(w));
//    for(int i=0;i<h;++i){
//        for(int j=0;j<w;++j){
//            chessboardCorners[i][j] = corners[i*w+j];
//        }
//    }

    //four corner point: [0, 0], [0, w-1], [h-1, w-1], [h-1, 0]
    int minCorner=0;
    float minValue = corners[0].x + corners[0].y;
    for(int i=0;i<h;i+=h-1){
        for(int j=0;j<w;j+=w-1){
            if(corners[i*w+j].x + corners[i*w+j].y < minValue){
                minCorner = i*w+j;
                minValue = corners[i*w+j].x + corners[i*w+j].y;
            }
        }
    }
    cout << "Left-top corner index: " << minCorner << endl;
    if (minCorner == (h -1) * w + w-1){ //rotate clockwise 180
        for(int i=0;i<corners.size()/2;++i){
            swap(corners[i], corners[corners.size()-1-i]);
        }
    } else if (minCorner == w - 1) { // w == h , rotate un-clockwise 90
        int n = w;
        for(int i=0;i<n;++i){
            for(int j=i+1;j<n;++j){
                swap(corners[i*n+j], corners[j*n+i]);
            }
        }
        for(int i=0;i<n/2;++i){
            for(int j=0;j<n;++j){
                swap(corners[i*n+j], corners[(n-1-i)*n+j]);
            }
        }
    } else if (minCorner == (h-1)*w + 0) { // w == h , rotate clockwise 90
        int n = w;
        for(int i=0;i<n;++i){
            for(int j=0;j<n-i-1;++j){
                swap(corners[i*n+j], corners[(n-1-j)*n + n - 1 - i]);
            }
        }
        for(int i=0;i<n/2;++i){
            for(int j=0;j<n;++j){
                swap(corners[i*n+j], corners[(n-1-i)*n+j]);
            }
        }
    }



    contour.reserve((w + h) * 2 - 4);
    for(int j=0;j<w;++j){
        contour.push_back(corners[j]);
    }
    for(int i=1;i<h;++i){
        contour.push_back(corners[i * w  + w - 1]);
    }
    for(int j=patternSize.width-2;j>=0;--j){
        contour.push_back(corners[(h-1)*w + j]);
    }
    for(int i=h-2;i>0;--i){
        contour.push_back(corners[i*w + 0]);
    }
    Moments M = moments(contour);
    Point2f center(M.m10/M.m00, M.m01/M.m00);
    //cout << center << endl;
    for(const Point2f& c:contour){
        double delta_x = c.x - center.x;
        double delta_y = c.y - center.y;
        double delta_len = sqrt(delta_x * delta_x + delta_y*delta_y);
        double udx = delta_x / delta_len;
        double udy = delta_y / delta_len;
        contour_ex.emplace_back(Point2i(round(c.x + udx * expandLen), round(c.y + udy * expandLen)));
    }
    return contour_ex;
}

void getProjectionMap(const Mat & cameraMatrix, const Mat&distCoeffs, const Mat&rvec, const Mat&tvec, const Size& targetSize, float pixLen, Mat& map_x, Mat& map_y){
    map_x = Mat(targetSize, CV_32F);
    map_y = Mat(targetSize, CV_32F);
    vector<Point3f>objPoints;
    vector<Point2f> imagePoints;
    calcChessboardCornerPositions(targetSize, pixLen, objPoints);
    projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
    for(int i=0;i<targetSize.height;++i){
        for(int j=0;j<targetSize.width;++j){
            map_x.at<float>(i, j) = imagePoints[i * targetSize.width + j].x;
            map_y.at<float>(i, j) = imagePoints[i * targetSize.width + j].y;
        }
    }
}

// 定位靶面上的棋盘格，确定相机外参
bool calcExParameters(const string &configFilePath, const Mat& frame, const string& outputFilePath,
                      int patternWidth, int patternHeight, float squareSize){
    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001 );

    Mat cameraMatrix, distCoeffs;
    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints(1);
    vector<Mat> rvecs, tvecs;
    vector<Point3f> newObjPoints;
    vector<Point2f> pointBuf;

    Size patternSize(patternWidth, patternHeight);
    int imageWidth, imageHeight;
    bool found;
    Mat img;
    Mat viewGray;

    Mat rot_mat;
    Mat exMatrix(4, 4, CV_64F);
    Mat inMatrix(3, 4, CV_64F);
    Mat B;

    Mat map_x, map_y;
    FileStorage fs;


    fs.open(configFilePath, FileStorage::READ);
    if(!fs.isOpened()){
        cout << "Calibration output file cannot open!" << endl;
        return false;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["image_width"] >> imageWidth;
    fs["image_height"] >> imageHeight;
    fs.release();


    img = frame.clone();
    if(img.empty()){
        cout << "Mat is empty!" << endl;
        return false;
    }

    cvtColor(img, viewGray, COLOR_BGR2GRAY);
//    found = checkChessboard(viewGray, patternSize);
//    if(!found){
//        cout << "First check failed. \nTry again. If no response please re-run the program." << endl;
//        // return false;
//    } else {
//        cout << "First check passed." << endl;
//    }

    found = findChessboardCorners(viewGray, patternSize, pointBuf, CALIB_CB_NORMALIZE_IMAGE);
    if(!found){
        cout << "Could not find chessboard!" << endl;
        return false;
    } else {
        cout << "Chessboard found." << endl;
    }
    vector<Point2i> contour = expandCorners(pointBuf, patternSize, 50);
    int winSize = 11;

    cornerSubPix(viewGray, pointBuf, Size(winSize,winSize), Size(-1,-1), criteria);

    imagePoints.push_back(pointBuf);
    int flags = CALIB_USE_INTRINSIC_GUESS+CALIB_FIX_K1+CALIB_FIX_K2+CALIB_FIX_K3+CALIB_FIX_K4+CALIB_FIX_K5;
    //flags = 0;

    calcChessboardCornerPositions(patternSize, squareSize, objectPoints[0]);
    objectPoints.resize(imagePoints.size(),objectPoints[0]);
    Size imageSize = img.size();

    double ret = calibrateCameraRO(objectPoints, imagePoints, imageSize,
                                   -1, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints, flags);

    Rodrigues(rvecs[0], rot_mat);

    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            exMatrix.at<double>(i, j) = rot_mat.at<double>(i, j);
            inMatrix.at<double>(i, j) = cameraMatrix.at<double>(i, j);
        }
        inMatrix.at<double>(i, 3) = 0;
        exMatrix.at<double>(i, 3) = tvecs[0].at<double>(i, 0);
        exMatrix.at<double>(3, i) = 0.;
    }
    exMatrix.at<double>(3, 3) = 1;
    dot(inMatrix, exMatrix, B);
    cout<<"RMS: "<<ret<<endl;

    fs.open(outputFilePath, FileStorage::WRITE_BASE64);
    fs << "location_params" << LocationParams(
            cameraMatrix, distCoeffs,
            exMatrix, B, rvecs[0], tvecs[0],
            contour);
    fs.release();
    return true;
}


void getBlackCenter(const Mat&gray, Point2f& center,double &areaSize, ObjectPointFinder& of){
    Mat binary;
    vector<vector<Point>>contours;
    vector<Vec4i>hierarchy;
    Point2f blackCenterOfImage;
    double *objPoint = of.retAddr();
    // 使用OSTU方法找到中间的黑色区域
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // 寻找最大的轮廓
    if(contours.size()>1){
        int mxi = 0;
        double mxa = 0;
        for(int i=0;i<contours.size();++i){
            if(contourArea(contours[i]) > mxa){
                mxi = i;
                mxa = contourArea(contours[i]);
            }
        }
        contours[0] = contours[mxi];
    }
//    Mat whiteMask = Mat(gray.size(), CV_8U, Scalar(0));
//    drawContours(whiteMask, contours, 0, 255, -1);
//    imshow("Camera View", whiteMask);
//    waitKey(1000);

    vector<Point2f>objContour;
    // 寻找图像轮廓对应的实际轮廓
    of.findPoints(contours[0], objContour);

    // 寻找黑色区域现实中的中心位置
    Moments M = moments(objContour);
    center = Point2f(M.m10/M.m00, M.m01/M.m00);
    areaSize = contourArea(objContour);
}

// return eight directions: 方向判定，把360°分成八个方向
wchar_t getDirection(double offset_x, double offset_y){
    double angle = atan2(offset_y, offset_x);
    wchar_t ds[]=L"☆→↗↑↖←↙↓↘";
    int index = 0;
    if(angle<0)angle += 2*PI;
    if(angle < PI/8 || angle >= PI*15/8){
        index = 1;
    } else if(angle>=PI/8 && angle < PI*3/8){
        index = 2;
    } else if(angle>=PI*3/8 && angle < PI*5/8){
        index = 3;
    } else if(angle>=PI*5/8 && angle < PI*7/8){
        index = 4;
    } else if(angle>=PI*7/8 && angle < PI*9/8){
        index = 5;
    } else if(angle>=PI*9/8 && angle < PI*11/8){
        index = 6;
    } else if(angle>=PI*11/8 && angle < PI*13/8){
        index = 7;
    } else if(angle>=PI*13/8 && angle < PI*15/8){
        index = 8;
    }
    // 如果接近中心，直接返回五角星
    if(sqrt(offset_x*offset_x + offset_y*offset_y) < 5){
        index = 0;
    }

    return ds[index];
}

// 计算走纸距离
double rollDistance(double offset_x, double offset_y){
    double R1 = 30 + 4.5;
    return sqrt(R1 * R1 - offset_x * offset_x) - offset_y;
}

// 间距0.8mm 或者 0.25mm（circle_distance=2.5)
// target_distance: 毫米单位*10
int calculateGrades(double target_distance, double circle_distance = 8){
    int res = min(109, (int)floor(110.0 - target_distance/circle_distance))/10;
    if(res < 10)res = 0;
    return res;
}


float distance(const Point2f& a, const Point2f& b){
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}

void toggleStatus(bool& status, const string& winName, int width, int height){
    if(status){
        destroyWindow(winName);
    } else{
        namedWindow(winName, WINDOW_FREERATIO);
        resizeWindow(winName, width, height);
    }
    status = !status;
}
//
//找轮廓
int directions[4][2]={{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
bool isDebug = true;
int findContoursByBFS(const Mat& blankBackground, const Mat& gray,
                      const DetectionSettings& detectionSettings,
                      vector<vector<Point>>&resContours){
    vector<Point>convex;
    resContours.clear();
    Mat visited(blankBackground.size(), CV_8U, Scalar(0));
    int R = blankBackground.rows;
    int C = blankBackground.cols;
    int u1, u2;
    for(int r = 0;r < R; r += 10) {
        for (int c = 0; c < C; c += 10) {
            if (visited.at<uchar>(r, c))
                continue;
            u1 = blankBackground.at<uchar>(r, c);
            u2 = gray.at<uchar>(r, c);
            if (u1 > u2 && u1 - u2 >= detectionSettings.grayDiffThreshold) {
                //BFS
                vector<Point> pts;
                queue<pair<int, int>> q;
                q.emplace(r, c);
                visited.at<uchar>(r, c) = 1;
                while (!q.empty()) {
                    pair<int, int> t = q.front();
                    q.pop();
                    pts.emplace_back(t.second, t.first);
                    if (pts.size() > detectionSettings.holePixesMax)break;
                    for (auto &d:directions) {
                        int nr = t.first + d[0];
                        int nc = t.second + d[1];
                        if (0 <= nr && nr < R && 0 <= nc && nc < C && !visited.at<uchar>(nr, nc)) {
                            u1 = blankBackground.at<uchar>(nr, nc);
                            u2 = gray.at<uchar>(nr, nc);
                            if (u1 > u2 && u1 - u2 >= detectionSettings.grayDiffThreshold) {
                                q.emplace(nr, nc);
                                visited.at<uchar>(nr, nc) = 1;
                            }
                        }
                    }
                }
                // 黑色区域像素数超过配置文件规定的最大值
                if (pts.size() > detectionSettings.holePixesMax) {
                    cout << "Dark area too large. Dispose current frame." << endl;
                    return -1;
                }
                // 黑色区域像素数介于最小值最大值之间，视为有效
                if (pts.size() > detectionSettings.holePixesMin) {
                    // 在最小矩形区域寻找图像点集对应的图像轮廓（取最大）
                    Rect roiRect = cv::boundingRect(pts);
                    roiRect.x = max(0, roiRect.x - 1);
                    roiRect.y = max(0, roiRect.y - 1);
                    roiRect.width = min(roiRect.width + 2, gray.cols - roiRect.x);
                    roiRect.height = min(roiRect.height + 2, gray.rows - roiRect.y);
                    Mat diff, binary;
                    vector<vector<Point>> contours;
                    subtract(blankBackground(roiRect), gray(roiRect), diff);
                    threshold(diff, binary, detectionSettings.grayDiffThreshold, 255, THRESH_BINARY);
                    findContours(binary, contours, noArray(), RETR_LIST, CHAIN_APPROX_SIMPLE, roiRect.tl());
                    //选取最大的轮廓
                    if(isDebug)
                    {
                        imshow("binary", binary);
                    }

                    if (!contours.empty()) {
                        //cout << "More than one contour. finding the biggest." << endl;
                        int max_area_index = 0;
                        double max_area = -1, current_area;
                        for (int i = 0; i < contours.size(); ++i) {
                            current_area = contourArea(contours[i]);
                            if (current_area > max_area) {
                                max_area_index = i;
                                max_area = current_area;
                            }
                        }
                        convexHull(contours[max_area_index], convex);
                        if(contourArea(convex) / contourArea(contours[max_area_index]) > 2){
                            cout << "Abandon: convex area too large. " << (contourArea(convex) / contourArea(contours[max_area_index])) << endl;
                            continue;
                        }

                        resContours.emplace_back(contours[max_area_index]);
                    } //end if of finding biggest contour
                } //end if of finding contour
            } // end if of searching
        } // end for cols
    } // end for rows
    return (int)resContours.size();
}

bool findHoughCircleByContour(ObjectPointFinder& of, const vector<Point>&contour,
                             const DetectionSettings& detectionSettings, Point2f& targetCenter){
    vector<Point2f>objContour;
    vector<Point>objPixContour;
    vector<Vec3f> circles;
    Rect objRoiRect;
    Mat objHoleContourMat;

    //将像素坐标转化为世界坐标
    of.findPoints(contour, objContour);
    //计算世界坐标的轮廓平铺到图像中的图像坐标
    objPixContour.reserve(objContour.size());
    for(Point2f&pf: objContour){
        objPixContour.emplace_back(
                cvRound(pf.x * (float)detectionSettings.pixesPerMM),
                cvRound(pf.y * (float)detectionSettings.pixesPerMM)
                );
    }
    //计算图像的长和宽、左上角的位置
    objRoiRect = cv::boundingRect(objPixContour);
    // 区域太长或者太宽，也舍去
    if(objRoiRect.width > detectionSettings.calcRadiusMax * 3 || objRoiRect.height > detectionSettings.calcRadiusMax * 3){
        cout << "Selected area too tall/wide." << endl;
        return false;
    }
    //将轮廓绘制到图像中，以左上角的位置作为偏移量
    objHoleContourMat = Mat(objRoiRect.height + 2, objRoiRect.width + 2, CV_8U, Scalar(0));
    drawContours(objHoleContourMat,
                 vector<vector<Point>>(1, objPixContour),
                 0, 255, -1, LINE_8, noArray(), 2147483647,
                 -objRoiRect.tl() + Point(1, 1));
    //查找霍夫圆
    HoughCircles(objHoleContourMat, circles, HOUGH_GRADIENT,
                 2, 100, 100, detectionSettings.holeThreshold,
                 detectionSettings.calcRadiusMin, detectionSettings.calcRadiusMax);
    if(circles.empty()){
        cout << "No hough circles found." << endl;
        return false;
    }else {
        //////////////调试阶段画出霍夫圆////////////////////
        Mat colorHoleImg;
        cvtColor(objHoleContourMat, colorHoleImg, COLOR_GRAY2BGR);
        Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
        circle(colorHoleImg, center, 3, Scalar(255,0,0), -1, 8, 0 );
        circle(colorHoleImg, center, cvRound(circles[0][2]), Scalar(0,0,255), 1, 8, 0 );
        imshow("hole", colorHoleImg);
        /////////////////////////////////////////////
        // 霍夫圆心加上偏移量作为输出的结果
        targetCenter = (Point2f(objRoiRect.tl() - Point(1, 1)) + Point2f(circles[0][0], circles[0][1]))/detectionSettings.pixesPerMM;
        return true;
    }
}

int findHolesByContours(const Mat& blankBackground, const Mat& gray,
                        const DetectionSettings& detectionSettings,
                        vector<vector<Point>>&resContours){
    Mat diff, binary;
    vector<vector<Point>>contours;
    // 图像相减，负值为零
    subtract(blankBackground, gray, diff);
    //normalize(diff, diff, 0, 256, NORM_MINMAX);
    threshold(diff, binary, detectionSettings.grayDiffThreshold, 255, THRESH_BINARY);
    //轮廓检测
    findContours(binary, contours, noArray(), RETR_LIST, CHAIN_APPROX_SIMPLE);

    for(const vector<Point>&pts: contours){

        vector<Point>convexOfPix;
        convexHull(pts, convexOfPix);

        // 判断凸包与轮廓面积比值， 如果太大说明轮廓中间空白，不是目标点，舍去
        if(contourArea(convexOfPix) / contourArea(pts) > 2){
            cout << "Abandon for 2st reason(convex): " << (contourArea(convexOfPix) / contourArea(pts)) << endl;
            continue;
        }
        resContours.emplace_back(pts);

//        if(arcLen*arcLen/area/4 > 4){
//            cout << "Abandon for 3rd reason(ratio): " << (arcLen*arcLen/area/4) << endl;
//            continue;
//        }
    }
    return (int)resContours.size();
}

void mouseHandler1(int event, int x, int y, int flags, void*param){
    if(event == cv::EVENT_MOUSEMOVE){
        auto*pt = (Point*)param;
        pt->x = x;
        pt->y = y;
    }
}

void ZoomInWindow(MyFrameCapture& cap){
    Mat frame, subImg;
    Point ptc;
    setMouseCallback("Camera View",mouseHandler1, &ptc);
    namedWindow("Sub Image", WINDOW_FREERATIO);
    resizeWindow("Sub Image", 602, 602);
    while(cap.isOpened()){
        cap.nextImage(frame);
        if(ptc.x - 150 < 0)ptc.x = 150;
        else if(ptc.x + 150 >= frame.cols)ptc.x = frame.cols - 1 - 150;

        if(ptc.y - 150 < 0) ptc.y = 150;
        else if(ptc.y + 150 >= frame.rows)ptc.y = frame.rows - 1 - 150;

        subImg = Mat(frame, Rect(ptc.x-150, ptc.y-150, 301, 301));
        imshow("Sub Image", subImg);
        rectangle(frame, Rect(ptc.x-150, ptc.y-150, 301, 301), Scalar(0, 0, 255), 4);
        imshow("Camera View", frame);
        int key = waitKey(30);
        if(key == 27){
            break;
        }
    }
    destroyWindow("Sub Image");
}









