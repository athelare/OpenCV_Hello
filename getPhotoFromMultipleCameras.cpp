//
// Created by Jiyu_ on 2021/8/1.
//

#include"include/DVPCamera.h"
#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;





int main4(){
    printf("start...\r\n");

    int cameraCount = 0;
    int captureCount = 0;

    const char ESC_KEY = 27;
    const char ENTER_KEY = 13;

    dvpUint32 count = 0, num = -1;
    dvpCameraInfo info[8];
    dvpHandle h[8];
    dvpFrame frames[8];
    dvpStatus status;
    void *p[8];

    char timeStr[20];
    char fileName[64];

    time_t seconds;
    tm *tm1 = new tm();

    /* 枚举设备 */
    dvpRefresh(&count);
    if (count > 8)
        count = 8;

    for (int i = 0; i < (int)count; i++)
    {
        if (dvpEnum(i, &info[i]) == DVP_STATUS_OK)
        {
            printf("[%d]-Camera FriendlyName : %s\r\n", i, info[i].FriendlyName);
            cameraCount++;
        }
    }

    /* 没发现设备 */
    if (count == 0)
    {
        printf("No device found!\n");
        return 0;
    }

    for(int i=0;i<cameraCount;++i){
        dvpRegion region;
        double exp;
        float gain;
        bool trigMode = false;

        status = dvpOpen(i, OPEN_NORMAL, &h[i]);
        if(status != DVP_STATUS_OK){
            printf("dvpOpenByName failed with err:%d\r\n", status);
            break;
        }
        status = dvpSetTriggerState(h[i], false);
        if (status == DVP_STATUS_OK)
        {
            trigMode = false;
        }
        else
        {
            printf("dvpSetTriggerState failed with err:%d\r\n", status);
            break;
        }

        /* 打印ROI信息 */
        status = dvpGetRoi(h[i], &region);
        if (status != DVP_STATUS_OK)
        {
            printf("dvpGetRoi failed with err:%d\r\n", status);
            break;
        }
        printf("%s, region: x:%d, y:%d, w:%d, h:%d\r\n", info[i].FriendlyName, region.X, region.Y, region.W, region.H);

        /* 打印曝光增益信息 */
        status = dvpGetExposure(h[i], &exp);
        if (status != DVP_STATUS_OK)
        {
            printf("dvpGetExposure failed with err:%d\r\n", status);
            break;
        }

        status = dvpGetAnalogGain(h[i], &gain);
        if (status != DVP_STATUS_OK)
        {
            printf("dvpGetAnalogGain failed with err:%d\r\n", status);
            break;
        }

        printf("%s, exposure: %lf, gain: %f\r\n", info[i].FriendlyName, exp, gain);


        /* 开始视频流 */
        status = dvpStart(h[i]);
        if (status != DVP_STATUS_OK)
        {
            break;
        }
    }

//    while (num < 0 || num >= count)
//    {
//        printf("Please enter the number of the camera you want to open: \r\n");
//        scanf("%d", &num);
//    }

//    thread task(test, (void*)info[num].FriendlyName);
//    task.join();

    while(true){
        for(int i=0;i<cameraCount;++i){
            status = dvpGetFrame(h[i], &frames[i], &p[i], 3000);
            cv::Mat img(frames[i].iHeight, frames[i].iWidth, CV_8UC3, p[i]);
            cv::resize(img, img, cv::Size(1224, 1024));
            cv::imshow(info[i].FriendlyName, img);
        }
        int key = cv::waitKey(30);
        if(key == ESC_KEY){
            break;
        }else if(key == ENTER_KEY){
            seconds = time(nullptr);
            localtime_s(tm1, &seconds);

            sprintf(timeStr, "%d%02d%02d_%02d%02d%02d",
                    tm1->tm_year+1900,
                    tm1->tm_mon+1,
                    tm1->tm_mday,
                    tm1->tm_hour,
                    tm1->tm_min,
                    tm1->tm_sec);
            for(int i=0;i<cameraCount;++i){
                sprintf(fileName, "capture/%d/%s.png", i, timeStr);
                status = dvpSavePicture(&frames[i], p[i], fileName, 100);
                if (status == DVP_STATUS_OK){
                    printf("%d %s OK\r\n",captureCount, fileName);
                }
            }
            captureCount++;
        }
    }
    for(int i=0;i<cameraCount;++i){
        status = dvpStop(h[i]);
        status = dvpClose(h[i]);
    }

    printf("quit.\n");
    return 0;
}