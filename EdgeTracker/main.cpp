//
//  main.cpp
//  EdgeTracker
//
//  Created by Daniel Mesham on 03/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "area.hpp"
#include "asm.hpp"
#include "lsq.hpp"
#include "models.hpp"
#include "orange.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace std;
using namespace cv;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Methods
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

void mouseHandler(int event, int x, int y, int flags, void* param);
void addMouseHandler(cv::String window);
string currentDateTime();

// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Constants
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// The intrinsic matrix: Mac webcam
static float intrinsicMatrix[3][3] = {
    { 1045.8,    0  , 646.7 },
    {    0  , 1058.8, 350.9 },
    {    0  ,    0  ,   1   }
};
static Mat K = Mat(3,3, CV_32FC1, intrinsicMatrix);

static string dataFolder = "../../../../../data/";
static string logFolder = "../../../../../logs/";

static bool LOGGING = false; // Whether to log data to CSV files
static bool REPORT_ERRORS = true; // Whether to report the area error (slows performance)
static bool USE_LINE_ITER = true; // Whether to use the line iterator technique for the whiskers



// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Main Method
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *


int main(int argc, const char * argv[]) {
    
    // * * * * * * * * * * * * * * * * *
    //   OPEN THE FIRST FRAME
    // * * * * * * * * * * * * * * * * *
    
    Mat frame;
    String filename = "ArrowDiamond_3.avi";
    VideoCapture cap(dataFolder + filename);
    //VideoCapture cap(0); waitKey(1000);   // Uncomment this line to try live tracking
    if(!cap.isOpened()) return -1;
    
    cap >> frame;
    imshow("Frame", frame);
    
    // * * * * * * * * * * * * * * * * *
    //   MODEL CREATION
    // * * * * * * * * * * * * * * * * *
    
    Model * modelRect = new Rectangle(60, 80, Scalar(20, 65, 165));
    Model * modelDog = new Dog(Scalar(19, 89, 64));
    Model * modelArrow = new Arrow(Scalar(108, 79, 28));
    Model * modelTriangle = new Triangle(Scalar(15, 0, 82));
    Model * modelDiamond = new Diamond(Scalar(13, 134, 161));
    Model * modelHouse = new House(Scalar(90, 90, 90));
    Model * modelYellowBox = new Box(175, 210, 49, Scalar(0, 145, 206));    // Yellow box
    Model * modelBrownBox = new Box(204, 257, 70, Scalar(71, 92, 121));     // Brown box
    Model * modelBlueBox = new Box(300, 400, 75, Scalar(180, 83, 40));      // Blue foam box
    Model * modelBrownCube = new Box(70, 70, 70, Scalar(35, 55, 90));       // Brown numbers cube

    vector<Model *> model;
    
    // * * * * * * * * * * * * * * * * *
    //   SELECT MODELS
    // * * * * * * * * * * * * * * * * *
    vector<estimate> est, prevEst;
    if (filename == "Test.avi") {
        model = {modelArrow, modelDiamond};
    }
    else if (filename == "Trio_1.avi") {
        model ={modelRect, modelDog, modelArrow};
        est = { estimate({ 50, -56, 363, -0.89, -0.15, -0.13}, 0, 0),
                estimate({ 18,  32, 243, -0.92, -0.02, -0.01}, 0, 0),
                estimate({-69, -29, 324, -0.89, -0.08, -0.05}, 0, 0) };
    }
    else if (filename == "Trio_2.avi") {
        model ={modelRect, modelDog, modelArrow};
        est = { estimate({ 80,   7, 273, -0.92,  0.02,  0.02}, 0, 0),
                estimate({-62, -15, 306, -0.92,  0.03,  0.03}, 0, 0),
                estimate({-34,  14, 270, -0.94,  0.02,  0.03}, 0, 0) };
    }
    else if (filename == "Trio_3.avi") {
        model ={modelRect, modelDog, modelArrow};
        est = { estimate({-69,   5, 268, -0.93,  0.05,  0.01}, 0, 0),
                estimate({ 21,  36, 251, -0.89, -0.09, -0.08}, 0, 0),
                estimate({ 29, -50, 357, -0.88,  0.01,  0.03}, 0, 0) };
    }
    else if (filename == "TrioHand_1.avi") {
        model ={modelRect, modelDog, modelArrow};
        est = { estimate({ 95,  43, 360, -0.80,  0.25,  0.05}, 0, 0),
                estimate({-77,  77, 311, -0.81,  0.11,  0.01}, 0, 0),
                estimate({ 50, -21, 413, -0.77,  0.14,  0.05}, 0, 0) };
    }
    else if (filename == "TrioHand_2.avi") {
        model ={modelRect, modelDog, modelArrow};
        est = { estimate({ 85,  28, 370,  0.16,  0.78,  1.65}, 0, 0),
                estimate({-86,  60, 322, -0.83,  0.06, -0.08}, 0, 0),
                estimate({ 28, -35, 430, -0.79,  0.05, -0.03}, 0, 0) };
    }
    else if (filename == "TrioHand_3.avi") {
        model ={modelRect, modelDog, modelArrow};
        est = { estimate({-60,   0, 393, -0.78,  0.17, -0.02}, 0, 0),
                estimate({ 18,  69, 328, -0.79,  0.11,  0.03}, 0, 0),
                estimate({ 64, -26, 422, -0.77,  0.08,  0.00}, 0, 0) };
    }
    else if (filename == "TrioHand_4.avi") {
        model ={modelRect, modelDog, modelArrow};
        est = { estimate({-44,  -6, 380, -0.85,  0.10,  0.05}, 0, 0),
                estimate({ 18, -13, 413, -0.81,  0.11,  0.02}, 0, 0),
                estimate({ 80,  27, 370, -0.76,  0.10,  0.04}, 0, 0) };
    }
    else if (filename == "RectDog_1.avi") {
        model ={modelRect, modelDog};
        est = { estimate({  45,  10, 339, -0.82, -0.01,  0.00}, 0, 0),
                estimate({ -92,  43, 299, -0.83, -0.04, -0.01}, 0, 0) };
    }
    else if (filename == "RectDog_2.avi") {
        model ={modelRect, modelDog};
        est = { estimate({  18,  11, 336, -0.67,  0.54,  0.58}, 0, 0),
                estimate({ -20,  -7, 352, -0.83, -0.03, -0.02}, 0, 0) };
    }
    else if (filename == "ArrowDiamond_1.avi") {
        model ={modelArrow, modelDiamond};
        est = { estimate({  56,   6, 339, -0.64,  0.53,  0.53}, 0, 0),
                estimate({ -77,  38, 294, -0.87, -0.07, -0.05}, 0, 0) };
    }
    else if (filename == "ArrowDiamond_2.avi") {
        model ={modelArrow, modelDiamond};
        est = { estimate({  -7, -58, 400,  0.13,  0.82,  1.77}, 0, 0),
                estimate({ -10, -10, 300, -2.30, -0.08,  0.02}, 0, 0) };
    }
    else if (filename == "ArrowDiamond_3.avi") {
        model ={modelArrow, modelDiamond};
        est = { estimate({  14, -61, 411,  0.77, -2.79,  0.38}, 0, 0),
                estimate({ -30,  41, 297, -0.86,  0.02,  0.04}, 0, 0) };
    }
    else if (filename == "BrownBox_1.avi") {
        model ={modelBrownBox};
        est = { estimate({-78,    8, 675, -0.83,  0.62,  0.14}, 0, 0) };
    }
    else if (filename == "BrownBox_2.avi") {
        model ={modelBrownBox};
        est = { estimate({-78,   10, 727, -0.99,  0.92,  0.55}, 0, 0) };
    }
    else if (filename == "BrownBox_BG_1.avi") {
        model ={modelBrownBox};
        est = { estimate({ 57,  -25, 456, -1.08, -0.27, -0.18}, 0, 0) };
    }
    else if (filename == "BrownBox_BG_2.avi") {
        model ={modelBrownBox};
        est = { estimate({ 43,  -30, 484, -0.78, -0.40, -0.45}, 0, 0) };
    }
    else if (filename == "BrownBox_BG_3.avi") {
        model ={modelBrownBox};
        est = { estimate({ 38,  -16, 490, -0.42,  0.84,  1.01}, 0, 0) };
    }
    else if (filename == "YellowBox_1.avi") {
        model ={modelYellowBox};
        est = { estimate({151,  -40, 590, -0.38,  0.77,  0.93}, 0, 0) };
    }
    else if (filename == "YellowBox_2.avi") {
        model ={modelYellowBox};
        est = { estimate({ 85,  -40, 635, -0.68,  0.43,  0.35}, 0, 0) };
    }
    else if (filename == "YellowBox_3.avi") {
        model ={modelYellowBox};
        est = { estimate({102,  -41, 520, -0.62, -0.32, -0.18}, 0, 0) };
    }
    else if (filename == "BlueYellow_1.avi") {
        model = {modelBlueBox,modelYellowBox};
        est = { estimate({  41,  26,  800, -0.85,  0.82,  0.53}, 0, 0),
                estimate({ -42, -38,  776, -0.98, -0.67, -0.40}, 0, 0) };
    }
    else if (filename == "BlueYellow_2.avi") {
        model = {modelBlueBox,modelYellowBox};
        est = { estimate({ -30, -72,  879, -0.70,  0.39,  0.83}, 0, 0),
                estimate({ 108, 105,  688, -0.25,  1.12,  1.27}, 0, 0) };
    }
    else if (filename == "BlueYellow_3.avi") {
        model = {modelBlueBox,modelYellowBox};
        est = { estimate({  28,  71,  905,  0.34,  1.19,  1.84}, 0, 0),
                estimate({  22,  46,  776, -1.14, -0.59, -0.30}, 0, 0) };
    }
    else if (filename == "BlueYellow_5.avi") {
        model = {modelBlueBox,modelYellowBox};
        est = { estimate({ 140,  63,  824, -1.17, -0.32, -0.12}, 0, 0),
                estimate({-195,  84,  802, -1.09, -0.68, -0.31}, 0, 0) };
    }
    else if (filename == "BlueYellow_7.avi") {
        model = {modelBlueBox,modelYellowBox};
        est = { estimate({-147,   6,  779, -1.24, -0.84, -0.25}, 0, 0),
                estimate({ 266,  15,  765, -1.35, -0.21, -0.04}, 0, 0) };
    }
    else if (filename == "BlueYellow_8.avi") {
        model = {modelBlueBox,modelYellowBox};
        est = { estimate({  18, -95,  798, -0.71, -0.81, -0.31}, 0, 0),
                estimate({-166,  56,  600, -1.27, -0.68, -0.19}, 0, 0) };
    }
    else if (filename == "BrownYellow_1.avi") {
        model = {modelBrownBox,modelYellowBox};
        est = { estimate({ 25,  -17, 559, -0.73,  0.48,  0.60}, 0, 0),
                estimate({ 25,  -75, 501, -0.70,  0.52,  0.60}, 0, 0) };
    }
    
    // * * * * * * * * * * * * * * * * *
    //   LOCATE THE STARTING POSITIONS
    // * * * * * * * * * * * * * * * * *
    //      Assume the objects are
    //      leaning back at
    //      ±45 degrees
    // * * * * * * * * * * * * * * * * *
    
    // If no initial poses are given, find them
    if (est.size() == 0) {
        // Find the initial pose of each model
        for (int m = 0; m < model.size(); m++) {
            
            // Find the area & centoid of the object in the image
            Mat segInit = orange::segmentByColour(frame, model[m]->colour);
            cvtColor(segInit, segInit, CV_BGR2GRAY);
            threshold(segInit, segInit, 0, 255, CV_THRESH_BINARY);
            Point centroid = ASM::getCentroid(segInit);
            double area = ASM::getArea(segInit);
            
            // Draw the model at the default position and find the area & cetroid
            Vec6f initPose = {0, 0, 300, -CV_PI/4, 0, 0};
            Mat initGuess = Mat::zeros(frame.rows, frame.cols, frame.type());
            model[m]->draw(initGuess, initPose, K, false);
            cvtColor(initGuess, initGuess, CV_BGR2GRAY);
            threshold(initGuess, initGuess, 0, 255, CV_THRESH_BINARY);
            Point modelCentroid = ASM::getCentroid(initGuess);
            double modelArea = ASM::getArea(initGuess);
            
            // Convert centroids to 3D/homogeneous coordinates
            Mat centroid2D;
            hconcat( Mat(centroid), Mat(modelCentroid), centroid2D );
            vconcat(centroid2D, Mat::ones(1, 2, centroid2D.type()), centroid2D);
            centroid2D.convertTo(centroid2D, K.type());
            Mat centroid3D = K.inv() * centroid2D;
            
            // Estimate the depth from the ratio of the model and measured areas,
            // and create a pose guess from that.
            // Note that the x & y coordinates need to be calculated using the pose
            // of the centroid relative to the synthetic model image's centroid.
            double zGuess = initPose[2] * sqrt(modelArea/area);
            centroid3D *= zGuess;
            initPose[0] = centroid3D.at<float>(0, 0) - centroid3D.at<float>(0, 1);
            initPose[1] = centroid3D.at<float>(1, 0) - centroid3D.at<float>(1, 1);
            initPose[2] = zGuess;
            
            // Set the intial pose
            estimate initEst = estimate(initPose, 0, 0);
            
            // Add the estimate to the list
            est.push_back(initEst);
        }
    }
    
    Mat frame2;
    frame.copyTo(frame2);
    for (int m = 0; m < model.size(); m++) {
        model[m]->draw(frame2, est[m].pose, K, true, model[m]->colour);
    }
    imshow("Frame", frame2);
    prevEst = est;
    
    // Uncomment to allow annotation
    //addMouseHandler("Frame");
    
    // * * * * * * * * * * * * * * * * *
    //   SET UP LOGGER
    // * * * * * * * * * * * * * * * * *
    ofstream log;
    if (LOGGING) {
        log.open(logFolder + currentDateTime() + ".csv");
        log << "Time";
        for (int m = 0; m < model.size(); m++) log << ";Model " << m;
        log << endl;
    }
    
    waitKey(0);
    
    // * * * * * * * * * * * * * * * * *
    //   CAMERA LOOP
    // * * * * * * * * * * * * * * * * *
    
    vector<double> times = {};
    double longestTime = 0.0;
    vector<vector<double>> errors = vector<vector<double>>(model.size());
    vector<double> worstError = vector<double>(model.size());
    
    while (!frame.empty()) {
        
        auto start = chrono::system_clock::now();   // Start the timer
        
        // Blur
        GaussianBlur(frame, frame, Size(3,3), 1);
        
        // Detect edges
        Mat canny, cannyTest;
        Canny(frame, canny, 20, 60);
        canny.copyTo(cannyTest);
        
        // Extract the image edge point coordinates
        Mat edges;
        if (!USE_LINE_ITER) findNonZero(canny, edges);
        else dilate(canny, canny, getStructuringElement(CV_SHAPE_CROSS, Size(3,3)));
        
        // Find the pose of each model
        for (int m = 0; m < model.size(); m++) {
            // Predict the next pose
            Vec6f posePrediction = est[m].pose + 0.5 * (est[m].pose - prevEst[m].pose);
            prevEst[m] = est[m];
            est[m].pose = posePrediction;
            
            int iterations = 1;
            double error = lsq::ERROR_THRESHOLD + 1;
            while (error > lsq::ERROR_THRESHOLD && iterations < 20) {
                // Generate a set of whiskers
                vector<Whisker> whiskers = ASM::projectToWhiskers(model[m], est[m].pose, K);
                
                // Sample along the model edges and find the edges that intersect each whisker
                Mat targetPoints = Mat(2, 0, CV_32S);
                Mat whiskerModel = Mat(4, 0, CV_32FC1);
                for (int w = 0; w < whiskers.size(); w++) {
                    Point closestEdge;
                    if (!USE_LINE_ITER) closestEdge = whiskers[w].closestEdgePoint(edges);
                    else closestEdge = whiskers[w].closestEdgePoint2(canny);
                    if (closestEdge == Point(-1,-1)) continue;
                    hconcat(whiskerModel, whiskers[w].modelCentre, whiskerModel);
                    hconcat(targetPoints, Mat(closestEdge), targetPoints);
                    
                    //TRACE: Display the whiskers
                    circle(cannyTest, closestEdge, 3, Scalar(120));
                    circle(cannyTest, whiskers[w].centre, 3, Scalar(255));
                    line(cannyTest, closestEdge, whiskers[w].centre, Scalar(150));
                    
                    //Point endPos = whiskers[w].centre + Point(40*whiskers[w].normal.x, 40*whiskers[w].normal.y);
                    //Point endNeg = whiskers[w].centre - Point(40*whiskers[w].normal.x, 40*whiskers[w].normal.y);
                    //line(cannyTest, endPos, endNeg, Scalar(200));
                }
                
                targetPoints.convertTo(targetPoints, CV_32FC1);
                
                // Catch error where no points are found
                if (whiskerModel.cols == 0) break;
                
                // Use least squares to match the sampled edges to each other
                est[m] = lsq::poseEstimateLM(est[m].pose, whiskerModel, targetPoints.t(), K, 2);
                
                
                double improvement = (error - est[m].error)/error;
                error = est[m].error;
                
                // Stop trying if you reduce the error by < 1% (excl. the first one)
                if (improvement < 0.01 && iterations > 1) break;
                
                iterations++;
                //waitKey(0);
            }
        }
        
        // Draw the shapes on the image
        for (int m = 0; m < model.size(); m++) {
             model[m]->draw(frame, est[m].pose, K, true);
        }
        imshow("Frame", frame);
        
        // Stop timer and show time
        auto stop = chrono::system_clock::now();
        chrono::duration<double> frameTime = stop-start;
        double time = frameTime.count()*1000.0;
        times.push_back(time);
        if (time > longestTime) longestTime = time;
        
        // Measure and report the area errors
        for (int m = 0; m < model.size(); m++) {
            if (REPORT_ERRORS) {
                Mat seg = orange::segmentByColour(frame, model[m]->colour);
                double areaError = area::areaError(est[m].pose, model[m], seg, K);
                errors[m].push_back(areaError);
                if (areaError > worstError[m]) worstError[m] = areaError;
            }
        }
        
        // Log time and errors
        if (LOGGING) {
            log << time;
            for (int m = 0; m < model.size(); m++) {
                log << ";" << errors[m].back();
            }
            log << endl;
        }
        
        //imshow("CannyTest", cannyTest);
        
        // Get next frame
        cap.grab();
        cap >> frame;
        
        if (waitKey(1) == 'q') break;
    }
    
    vector<double> meanTime, stdDevTime;
    meanStdDev(times, meanTime, stdDevTime);
    
    cout << "No. frames   = " << times.size() << endl;
    cout << "Avg time     = " << meanTime[0] << " ms     " << 1000.0/meanTime[0] << " fps" << endl;
    cout << "stdDev time  = " << stdDevTime[0] << " ms" << endl;
    cout << "Longest time = " << longestTime << " ms     " << 1000.0/longestTime << " fps" << endl;
    
    // Report errors
    if (REPORT_ERRORS) {
        cout << endl << "AREA ERRORS:" << endl << "Model   Mean     StDev    Worst" << endl;
        for (int m = 0; m < model.size(); m++) {
            vector<double> meanError, stdDevError;
            meanStdDev(errors[m], meanError, stdDevError);
            printf("%4i    %5.2f    %5.2f    %5.2f \n", m, meanError[0], stdDevError[0], worstError[m]);
            //cout << m << "   " << meanError[0] << "   " << stdDevError[0] << "   " << worstError[m] << endl;
        }
    }
    
    if (LOGGING) log.close();
    
    return 0;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Method Implementations
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

void mouseHandler(int event, int x, int y, int flags, void* param) {
    switch(event){
        case CV_EVENT_LBUTTONUP:
            cout << x << " " << y << endl;
            break;
    }
}

void addMouseHandler(cv::String window) {
    int mouseParam = CV_EVENT_FLAG_LBUTTON;
    setMouseCallback(window, mouseHandler, &mouseParam);
}

string currentDateTime() {
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    
    ostringstream oss;
    oss << put_time(&tm, "%Y.%m.%d_%H.%M.%S");
    return oss.str();
}
