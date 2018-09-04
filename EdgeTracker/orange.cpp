//
//  orange.cpp
//  LiveTracker
//
//  Created by Daniel Mesham on 01/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "orange.hpp"

using namespace std;
using namespace cv;


vector<Vec4i> orange::borderLines(Mat img) {
    // Returns the lines corresponding to the edges of the rectangle from an image.
    
    // Resolutions of the rho and theta parameters of the lines
    double res_rho = 5;
    double res_theta = CV_PI/180;
    int threshold = 60;
    double minLineLength = 50;
    double maxLineGap = 30;
    
    // Edge detection
    Mat dst;
    Canny(img, dst, 70, 210, 3, true);
    
    // Line detection
    vector<Vec4i> lines;
    while ((lines.size() < 4) && (threshold > 0)) {
        HoughLinesP(dst, lines, res_rho, res_theta, threshold, minLineLength, maxLineGap);
        threshold -= 5;
    }
    
    return lines;
}


Mat orange::segmentByColour(Mat img, Scalar colour) {
    
    // Convert colour to HSV
    Mat bgr(1 ,1 , CV_8UC3, colour);
    Mat3b hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Vec3b hsvPixel(hsv.at<Vec3b>(0,0));
    
    // Establish H, S, V ranges
    int thr[3] = {30, 50, 50};
    Scalar minHSV = Scalar(hsvPixel.val[0] - thr[0], hsvPixel.val[1] - thr[1], hsvPixel.val[2] - thr[2]);
    Scalar maxHSV = Scalar(hsvPixel.val[0] + thr[0], hsvPixel.val[1] + thr[1], hsvPixel.val[2] + thr[2]);
    
    // * * * * * * * * * *
    //      Blur & Sharpen
    // * * * * * * * * * *
    
    Mat blurred;
    int k = 3;
    GaussianBlur(img, blurred, Size(k,k), 1);
    
    // * * * * * * * * * *
    //      Segment
    // * * * * * * * * * *
    
    Mat imgHSV, imgMask, imgResult;
    cvtColor(blurred, imgHSV, COLOR_BGR2HSV);
    inRange(imgHSV, minHSV, maxHSV, imgMask);
    bitwise_and(img, img, imgResult, imgMask);
    
    //return imgResult;
    
    // * * * * * * * * * *
    //      Open/Close
    // * * * * * * * * * *
    
    k = 1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2*k + 1, 2*k + 1), Point(k, k));
    morphologyEx(imgResult, imgResult, MORPH_OPEN, kernel, Point(-1, -1), 1);
    morphologyEx(imgResult, imgResult, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    
    return imgResult;
}
