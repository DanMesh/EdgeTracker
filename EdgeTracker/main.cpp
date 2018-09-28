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
#include <chrono>

using namespace std;
using namespace cv;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Methods
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

void mouseHandler(int event, int x, int y, int flags, void* param);
void addMouseHandler(cv::String window);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Constants
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// The intrinsic matrix: Mac webcam
static float intrinsicMatrix[3][3] = {
    { 1047.7,    0  , 548.1 },
    {    0  , 1049.2, 362.9 },
    {    0  ,    0  ,   1   }
};
static Mat K = Mat(3,3, CV_32FC1, intrinsicMatrix);

static string dataFolder = "../../../../../data/";

static bool REPORT_ERRORS = true; // Whether to report the area error (slows performance)
static bool USE_LINE_ITER = true; // Whether to use the line iterator technique for the whiskers



// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Main Method
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *


int main(int argc, const char * argv[]) {
    
    // * * * * * * * * * * * * * * * * *
    //   MODEL CREATION
    // * * * * * * * * * * * * * * * * *
    
    Model * modelRect = new Rectangle(60, 80, Scalar(20, 65, 165));
    Model * modelDog = new Dog(Scalar(19, 89, 64));
    Model * modelArrow = new Arrow(Scalar(108, 79, 28));
    Model * modelYellowBox = new Box(175, 210, 49, Scalar(0, 145, 206));    // Yellow box
    Model * modelBrownBox = new Box(204, 257, 70, Scalar(71, 92, 121));     // Brown box
    Model * modelBlueBox = new Box(300, 400, 75, Scalar(180, 83, 40));      // Blue foam box
    Model * modelDogLive = new Dog(Scalar(75, 140, 85));                    // For use in live tracking
    
    vector<Model *> model = {modelRect, modelDog, modelArrow};
    //vector<Model *> model = {modelBlueBox};
    //vector<Model *> model = {modelDog};
    
    // * * * * * * * * * * * * * * * * *
    //   OPEN THE FIRST FRAME
    // * * * * * * * * * * * * * * * * *
    
    Mat frame;
    String filename = "Trio_1.avi";
    VideoCapture cap(dataFolder + filename);
    //VideoCapture cap(0); waitKey(1000);   // Uncomment this line to try live tracking
    if(!cap.isOpened()) return -1;
    
    cap >> frame;
    imshow("Frame", frame);
    
    // * * * * * * * * * * * * * * * * *
    //   LOCATE THE STARTING POSITIONS
    // * * * * * * * * * * * * * * * * *
    //      Assume the objects are
    //      leaning back at
    //      ±45 degrees
    // * * * * * * * * * * * * * * * * *
    
    vector<estimate> est, prevEst;
    
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
    prevEst = est;
    
    // Pause to allow annotation
    // addMouseHandler("Frame");
    
    
     // Uncomment block to try boxes with initialisation
    /*
    Mat frame2;
    frame.copyTo(frame2);
    //Vec6f posey = {40, 10, 810, -1.15, 0.18, -0.03}; // Blue_1.avi
    //Vec6f posey = {60, 35, 800, -1.15, 0.13, -0.01}; // Blue_2.avi
    Vec6f posey = {70, 0, 770, -1.27, 0.18, -0.02}; // Blue_3.avi
    //Vec6f posey = {-5, -7.3, 450, -1.02, -0.20, -0.09}; // Yellow_1.avi
    //Vec6f posey = {0, -24, 670, -1.02, 0.10, 0}; // Yellow_2.avi
    model[0]->draw(frame2, posey, K, true);
    est[0].pose = posey;
    prevEst = est;
    imshow("Frame", frame2);
    */
    
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
                
                // Use least squares to match the sampled edges to each other
                est[m] = lsq::poseEstimateLM(est[m].pose, whiskerModel, targetPoints.t(), K, 2);
                
                
                double improvement = (error - est[m].error)/error;
                error = est[m].error;
                
                // Stop trying if you reduce the error by < 1% (excl. the first one)
                if (improvement < 0.01 && iterations > 1) break;
                
                iterations++;
                //waitKey(0);
            }
            cout << "Iterations = " << iterations << endl;
        }
        
        // Stop timer and show time
        auto stop = chrono::system_clock::now();
        chrono::duration<double> frameTime = stop-start;
        double time = frameTime.count()*1000.0;
        cout << "Frame Time = " << time << " ms" << endl << endl;
        times.push_back(time);
        if (time > longestTime) longestTime = time;
        
        // Draw the shapes on the image
        for (int m = 0; m < model.size(); m++) {
            // Measure and report the area error
            if (REPORT_ERRORS) {
                Mat seg = orange::segmentByColour(frame, model[m]->colour);
                double areaError = area::areaError(est[m].pose, model[m], seg, K);
                errors[m].push_back(areaError);
                if (areaError > worstError[m]) worstError[m] = areaError;
            }
            
            model[m]->draw(frame, est[m].pose, K, true);
        }
        imshow("Frame", frame);
        
        imshow("CannyTest", cannyTest);
        
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
    if (!REPORT_ERRORS) return 0;
    cout << endl << "AREA ERRORS:" << endl << "Model   Mean     StDev    Worst" << endl;
    for (int m = 0; m < model.size(); m++) {
        vector<double> meanError, stdDevError;
        meanStdDev(errors[m], meanError, stdDevError);
        printf("%4i    %5.2f    %5.2f    %5.2f \n", m, meanError[0], stdDevError[0], worstError[m]);
        //cout << m << "   " << meanError[0] << "   " << stdDevError[0] << "   " << worstError[m] << endl;
    }
    
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
