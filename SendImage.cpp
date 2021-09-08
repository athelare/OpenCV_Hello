#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <ctime>
#include "main.h"
#define CURL_STATICLIB

using namespace std;
using namespace cv;

int SendImage(){
    Mat img = imread("capture/normal/20210829_170101_shot_9.0-(-4.4369,15.3319)-15.961.jpg");
    vector<uchar> buffer;
    imencode(".jpg", img, buffer);

    curl_global_init(CURL_GLOBAL_ALL);
    CURL* curl = curl_easy_init();

    curl_easy_setopt(curl,CURLOPT_URL,"http://127.0.0.1:8000/upload-picture?last-name=background&display=1");
    curl_easy_setopt(curl,CURLOPT_POST,1);
    curl_easy_setopt(curl,CURLOPT_POSTFIELDS, buffer.data());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, buffer.size());
    cout << "Sending request" << endl;
    clock_t st = clock();
    curl_easy_perform(curl);
    clock_t ed = clock();
    cout << (double)(ed - st) / CLOCKS_PER_SEC << endl;
    return 0;
}