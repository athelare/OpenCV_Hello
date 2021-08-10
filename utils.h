//
// Created by Jiyu_ on 2021/4/9.
//
#ifndef CV_TARGET_VISION_UTILS_H
#define CV_TARGET_VISION_UTILS_H

#define CURL_STATICLIB

#include <opencv2/opencv.hpp>
#include <cmath>
#include <thread>

#if defined(__linux__)
#include <wiringPi.h>
#endif
using namespace cv;
using namespace std;

#define PI 3.14159265358979323846

#if defined(__linux__)
class MotorController{
    const static int PUL=1;
    const static int DIR=4;
    const static int OFF=5;
public:
    static void Init(){
        cout << "wiringPi init: " << (wiringPiSetup()==0?"Success.":"Failed.") <<  endl;

        pinMode(OFF, OUTPUT);
        pinMode(PUL, OUTPUT);
        pinMode(DIR, OUTPUT);

        digitalWrite(DIR, HIGH);
    }
    static void Roll(double mm){
        cout << "Roll func called. ";
        int pulls = (mm*50/18);
        cout << "Total " << pulls << "pulls." << endl;
        while(pulls--){
            digitalWrite(PUL, HIGH); delay(1);
            digitalWrite(PUL, LOW); delay(1);
        }
        cout << "Roll func returned." << endl;
    }
    static void Ready(){
        digitalWrite(OFF, LOW);
    }
    static void Release(){
        digitalWrite(OFF, HIGH);
    }
};
#endif

class LocationParams{
private:
    Mat cameraMatrix, exMatrix, distCoeffs, B, rvec, tvec; // 相机标定参数
    vector<Point> backgroundContour;
public:
    LocationParams()= default;
    LocationParams::LocationParams(Mat _cameraMatrix, Mat _distCoeffs, Mat _exMatrix, Mat b, Mat _rvec, Mat _tvec, vector<Point> _backgroundContour){
        cameraMatrix = std::move(_cameraMatrix);
        distCoeffs = std::move(_distCoeffs);
        exMatrix = std::move(_exMatrix);
        B = std::move(b);
        rvec = std::move(_rvec);
        tvec = std::move(_tvec);
        backgroundContour = std::move(_backgroundContour);
    }
    Mat getCameraMatrix() const {return cameraMatrix;}
    Mat getExMatrix() const {return exMatrix;}
    Mat getDistCoeffs() const {return distCoeffs;}
    Mat getB() const {return B;}
    Mat getRvec() const {return rvec;}
    Mat getTvec() const {return tvec;}
    bool validate(){
        if(cameraMatrix.empty()){
            cout << "Camera matrix is empty!" << endl;
            return false;
        }
        if(exMatrix.empty()){
            cout << "Extrinsic matrix is empty!" << endl;
            return false;
        }
        if(distCoeffs.empty()){
            cout << "Distortion coefficients is empty!" << endl;
            return false;
        }
        if(B.empty()){
            cout << "B matrix is empty!" << endl;
            return false;
        }
        if(rvec.empty()){
            cout << "Rotation vector is empty!" << endl;
            return false;
        }
        if(tvec.empty()){
            cout << "Transformation vector is empty!" << endl;
            return false;
        }
        if(backgroundContour.empty()){
            cout << "Background contour is empty!" << endl;
            return false;
        }
        return true;
    }
    friend void read(const FileNode&node, LocationParams&x, const LocationParams& default_value){
        if(!node.empty()){
            node["camera_matrix"] >> x.cameraMatrix;
            node["distortion_coefficients"] >> x.distCoeffs;
            node["extrinsic_matrix"] >> x.exMatrix;
            node["b_matrix"] >> x.B;
            node["rvec"] >> x.rvec;
            node["tvec"] >> x.tvec;
            node["background_contour"] >> x.backgroundContour;
        }
    }
    friend void write(FileStorage& fs, const string&, const LocationParams& x){
        fs  << "{"
            << "camera_matrix" << x.cameraMatrix
            << "distortion_coefficients" << x.distCoeffs
            << "extrinsic_matrix" << x.exMatrix
            << "b_matrix" << x.B
            << "rvec" << x.rvec
            << "tvec" << x.tvec
            << "background_contour" << x.backgroundContour
            << "}";

    }
};

class DetectionSettings{
public:
    int grayDiffThreshold;
    int pixesPerMM;
    int holePixesMin;
    int holePixesMax;
    float holeRadiusMin;
    float holeRadiusMax;
    float validateThreshold;
    int holeThreshold;
    // 霍夫圆检测时输入的最小半径和最大半径
    int calcRadiusMin;
    int calcRadiusMax;
    void show() const{
        cout << "Detection settings:" << endl
            << "Gray differential threshold: " << grayDiffThreshold << endl
            << "Validate threshold: " << validateThreshold << endl
            << "Pixes per mm: " << pixesPerMM << endl
            << "Hole Pixes: " << holePixesMin << " to " << holePixesMax << endl
            << "Hole Radius: " << holeRadiusMin << " to " << holeRadiusMax << endl
            << "Hole threshold: " << holeThreshold << endl << endl;
    }
    bool validate() const{
        if(grayDiffThreshold == 0 || pixesPerMM == 0 || holePixesMin == 0 || holePixesMax == 0 ||
           holeThreshold == 0 )return false;
        return true;
    }
    friend void read(const FileNode&node, DetectionSettings&x, const DetectionSettings& default_value){
        if(!node.empty()) {
            node["gray_diff_threshold"] >> x.grayDiffThreshold;
            node["validate_threshold"] >> x.validateThreshold;
            node["pixes_per_mm"] >> x.pixesPerMM;
            node["hole_pixes_min"] >> x.holePixesMin;
            node["hole_pixes_max"] >> x.holePixesMax;
            node["hole_radius_min"] >> x.holeRadiusMin;
            node["hole_radius_max"] >> x.holeRadiusMax;
            node["hole_threshold"] >> x.holeThreshold;

            x.calcRadiusMin = int(x.holeRadiusMin * (float)x.pixesPerMM);
            x.calcRadiusMax = int(x.holeRadiusMax * (float)x.pixesPerMM);
        }
    }
};

class MainSettings{
private:
    string locationParamFile;
    string calibrationOutFile;
    string cameraIndex;
    string serverAddr;
    string identity;
    DetectionSettings detectionSettings;
public:
    string getLocationParamFile(){return locationParamFile;}
    string getCalibrationOutFile(){return calibrationOutFile;}
    string getCameraIndex(){return cameraIndex;}
    string getServerAddr(){return serverAddr;}
    string getIdentityStr(){return identity;}
    DetectionSettings& getDetectionSettings(){return detectionSettings;}
    bool validate(){
        if(locationParamFile.empty()){
            cout << "No puzzle file in config!" << endl;
            return false;
        }

        if(calibrationOutFile.empty()){
            cout << "No calibration output file in config!" << endl;
            return false;
        }

        if(serverAddr.empty()){
            cout << "No server host address in config!" << endl;
            return false;
        }

        if(identity.empty()){
            cout << "No identity string in config!" << endl;
            return false;
        }

        return true;
    }
    void show(){
        cout << endl
             << "Configurations from config file:"
             << "Puzzle file: " << locationParamFile << endl
             << "Calib  file: " << calibrationOutFile << endl
             << "Video input: " << cameraIndex << endl
             << "Server addr: " << serverAddr << endl
             << "Identity id: " << identity << endl
             << endl;
        detectionSettings.show();
    }
    friend void read(const FileNode&node, MainSettings&x, const MainSettings& default_value){
        if(!node.empty()) {
            node["puzzle_file"] >> x.locationParamFile;
            node["calibration_out_file"] >> x.calibrationOutFile;
            node["video_input"] >> x.cameraIndex;
            node["server_addr"] >> x.serverAddr;
            node["identity"] >> x.identity;
            node["detection_settings"] >> x.detectionSettings;
        }
    }
};


class ImageFileWriterTool{
    time_t seconds{};
    tm *tm1;
    char tmpStr[100]{};
public:
    ImageFileWriterTool(){
        tm1 = new tm();
    }
    ~ImageFileWriterTool(){
        delete tm1;
    }
    void writeFrame(const char* prefix, const Mat& _frame){
        seconds = time(nullptr);
#if defined(__linux__)
        localtime_r(&seconds, tm1);
#elif defined(_WIN32)
        localtime_s(tm1, &seconds);
#endif
        sprintf(tmpStr, "%s%d%02d%02d_%02d%02d%02d.jpg",
                prefix,
                tm1->tm_year+1900,
                tm1->tm_mon+1,
                tm1->tm_mday,
                tm1->tm_hour,
                tm1->tm_min,
                tm1->tm_sec);
        imwrite(tmpStr, _frame);
    }
};



class ObjectPointFinder{
    Mat exMatrix, B;
    double fx, fy, ux, uy, k1, k2, p1, p2, k3, k4, k5, k6;
    vector<vector<double>>equation;
    double correctedPoint[2]{0, 0};
    double ret[2]{0, 0};

    void undistortPoints(double x, double y)
    {
        //首先进行坐标转换；
        double xDistortion = (x - ux) / fx;
        double yDistortion = (y - uy) / fy;

        double xCorrected=0, yCorrected=0;

        double x0 = xDistortion;
        double y0 = yDistortion;
        //这里使用迭代的方式进行求解，因为根据2中的公式直接求解是困难的，所以通过设定初值进行迭代，这也是OpenCV的求解策略；
        for (int j = 0; j < 10; j++)
        {
            double r2 = xDistortion*xDistortion + yDistortion*yDistortion;

            double distRadialA = 1 / (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
            // double distRadialB = 1. + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
            double distRadialB = 1.;
            double deltaX = 2. * p1 * xDistortion * yDistortion + p2 * (r2 + 2. * xDistortion * xDistortion);
            double deltaY = p1 * (r2 + 2. * yDistortion * yDistortion) + 2. * p2 * xDistortion * yDistortion;

            xCorrected = (x0 - deltaX)* distRadialA * distRadialB;
            yCorrected = (y0 - deltaY)* distRadialA * distRadialB;

            xDistortion = xCorrected;
            yDistortion = yCorrected;
        }
        //进行坐标变换；
        xCorrected = xCorrected * fx + ux;
        yCorrected = yCorrected * fy + uy;

        correctedPoint[0] = xCorrected;
        correctedPoint[1] = yCorrected;

    }
public:
    double *retAddr(){return ret;}
    ObjectPointFinder(const Mat &cameraMatrix, const Mat& distCoeffs, const Mat& exMatrix, const Mat& B){

        equation = vector<vector<double>>(2, vector<double>(4));

        fx = cameraMatrix.at<double>(0, 0);
        fy = cameraMatrix.at<double>(1, 1);
        ux = cameraMatrix.at<double>(0, 2);
        uy = cameraMatrix.at<double>(1, 2);

        k1 = distCoeffs.at<double>(0, 0);
        k2 = distCoeffs.at<double>(1, 0);
        p1 = distCoeffs.at<double>(2, 0);
        p2 = distCoeffs.at<double>(3, 0);
        k3 = distCoeffs.at<double>(4, 0);
        k4 = k5 = k6 = 0;

        this->exMatrix = exMatrix.clone();
        this->B = B.clone();
    }
    void findPoint(double x, double y){
        undistortPoints(x, y);
        // cout<<"("<<correctedPoint[0]<<", "<<correctedPoint[1]<<") ->";
        double D, D1, D2;
        for(int i=0;i<2;++i){
            for(int j=0;j<4;++j){
                equation[i][j] = correctedPoint[i] * exMatrix.at<double>(2, j) - B.at<double>(i, j);
            }
        }

        equation[0][3]=-equation[0][3];
        equation[1][3]=-equation[1][3];

        D = equation[0][0]*equation[1][1] - equation[1][0]*equation[0][1];
        D1 = equation[0][3]*equation[1][1] - equation[1][3]*equation[0][1];
        D2 = equation[0][0]*equation[1][3] - equation[1][0]*equation[0][3];

        ret[0] = D1/D;
        ret[1] = D2/D;
    }
    void findPoints(const vector<Point>& pts, vector<Point2f>&res){
        res.clear();
        for(const auto& pt:pts){
            findPoint(pt.x, pt.y);
            res.emplace_back(ret[0], ret[1]);
        }
    }

};

class MyFrameCapture{
    enum CapType{CAPTURE, IMAGE}capType=CAPTURE;
    VideoCapture cap;
    Mat lookUpTable;
    Mat img;
    float exposure=0;
    int cameraIndex=0;
public:
    MyFrameCapture() = default;
    explicit MyFrameCapture(const string& cameraStr){
        if(isdigit(cameraStr[0])){
            capType = CAPTURE;
            cameraIndex = cameraStr[0] - '0';
            cap = VideoCapture(cameraIndex);
            cap.set(CAP_PROP_FRAME_WIDTH, 10000);
            cap.set(CAP_PROP_FRAME_HEIGHT, 10000);
            cout << "MyFrameCapture's frame size: "
                 << cap.get(CAP_PROP_FRAME_WIDTH) <<" x "
                 <<cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
            cap.grab();
        }else{
            capType = IMAGE;
            img = imread(cameraStr);
        }
        //Gamma correction
        lookUpTable = Mat(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        for( int i = 0; i < 256; ++i)
            p[i] = saturate_cast<uchar>(pow(i / 255.0, 0.6) * 255.0);
    }

    ~MyFrameCapture(){
        if(cap.isOpened())cap.release();
        img.release();
    }
    VideoCapture& getCap(){
        return cap;
    }
    bool isOpened(){
        if(capType == IMAGE){
            return !img.empty();
        } else{
            return cap.isOpened();
        }
    }

    bool nextImage(Mat& dst){
        if(capType == IMAGE){
            dst = img.clone();
        } else{
            //mutexLock->lock();
            for(int i=0;i<5;++i){
                cap.grab();
            }
            cap.retrieve(dst);
            //mutexLock->unlock();
        }
        return true;
    }
    bool nextGammaImage(Mat &dst){
        this->nextImage(img);
        LUT(img, lookUpTable, dst);
        return true;

    }
    void release(){
        if(capType == CAPTURE){
            cap.release();
        }
    }
    void reOpen(){
        if(capType == CAPTURE && !cap.isOpened()){
            cap.open(cameraIndex);
            cap.set(CAP_PROP_FRAME_WIDTH, 10000);
            cap.set(CAP_PROP_FRAME_HEIGHT, 10000);
            cout << "MyFrameCapture's frame size: "
                 << cap.get(CAP_PROP_FRAME_WIDTH) <<" x "
                 <<cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        }
    }
    double getExposure(){
        if(capType == CAPTURE){
            return cap.get(CAP_PROP_EXPOSURE);
        }else{
            return 0;
        }
    }
    void callCameraSettings(){
        if(capType == CAPTURE){
            cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);
            cap.set(CAP_PROP_SETTINGS, 0);
        }
    }
    void adjustExposureValue(int direction){
        int adjustStep=1;
#if defined(__linux__)
        adjustStep = 200;
#endif
        double currentValue = cap.get(CAP_PROP_EXPOSURE);
        if(direction == 1) {
            cap.set(CAP_PROP_EXPOSURE, currentValue + adjustStep);
        } else if(direction == -1) {
            cap.set(CAP_PROP_EXPOSURE, currentValue - adjustStep);
        }
    }
};


enum ScanMode{
    THRESHOLD,
    CANNY
};

bool calcExParameters(const string &configFilePath, const Mat& frame, const string& outputFilePath,
                      int patternWidth, int patternHeight, float squareSize);
void ZoomInWindow(MyFrameCapture& cap);


#endif //CV_TARGET_VISION_UTILS_H
