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



// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Main Method
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *


int main(int argc, const char * argv[]) {
    
    // * * * * * * * * * * * * * * * * *
    //   MODEL CREATION
    // * * * * * * * * * * * * * * * * *
    
    //Model * model = new Rectangle(60, 80, Scalar(20, 65, 165));
    Model * model = new Dog(Scalar(19, 89, 64));
    //Model * model = new Arrow(Scalar(108, 79, 28));
    //Model * model = new Box(175, 210, 49, Scalar(8, 152, 206)); // Yellow box
    //Model * model = new Box(204, 257, 70, Scalar(71, 92, 121)); // Brown box
    
    // * * * * * * * * * * * * * * * * *
    //   OPEN THE FIRST FRAME
    // * * * * * * * * * * * * * * * * *
    
    Mat frame;
    String filename = "Trio_1.avi";
    VideoCapture cap(dataFolder + filename);
    if(!cap.isOpened()) return -1;
    
    cap >> frame;
    imshow("Frame", frame);
    
    // * * * * * * * * * * * * * * * * *
    //   LOCATE THE STARTING POSITION
    // * * * * * * * * * * * * * * * * *
    //      Assume the object is
    //      leaning back at
    //      ±45 degrees
    // * * * * * * * * * * * * * * * * *
    
    // Find the area & centoid of the object in the image
    Mat segInit = orange::segmentByColour(frame, model->colour);
    cvtColor(segInit, segInit, CV_BGR2GRAY);
    threshold(segInit, segInit, 0, 255, CV_THRESH_BINARY);
    Point centroid = ASM::getCentroid(segInit);
    double area = ASM::getArea(segInit);
    
    // Draw the model at the default position and find the area & cetroid
    Vec6f pose = {0, 0, 300, -CV_PI/4, 0, 0};
    Mat initGuess = Mat::zeros(frame.rows, frame.cols, frame.type());
    model->draw(initGuess, pose, K);
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
    double zGuess = pose[2] * sqrt(modelArea/area);
    centroid3D *= zGuess;
    pose[0] = centroid3D.at<float>(0, 0) - centroid3D.at<float>(0, 1);
    pose[1] = centroid3D.at<float>(1, 0) - centroid3D.at<float>(1, 1);
    pose[2] = zGuess;
    waitKey(0);
    
    // Set the intial pose
    estimate est = estimate(pose, 0, 0);
    
    // Pause to allow annotation
    // addMouseHandler("Frame"); waitKey(0);
    
    // * * * * * * * * * * * * * * * * *
    //   CAMERA LOOP
    // * * * * * * * * * * * * * * * * *
    
    vector<double> times = {};
    double longestTime = 0.0;
    
    while (!frame.empty()) {
        
        auto start = chrono::system_clock::now();   // Start the timer
                
        // Segment by colour
        Mat seg = orange::segmentByColour(frame, model->colour);
        
        // Detect edges
        Mat canny;
        Canny(seg, canny, 70, 210);
        
        // Extract the image edge point coordinates
        Mat edges;
        findNonZero(canny, edges);
        
        int iterations = 1;
        double error = lsq::ERROR_THRESHOLD + 1;
        while (error > lsq::ERROR_THRESHOLD && iterations < 20) {
            // Generate a set of whiskers
            vector<Whisker> whiskers = ASM::projectToWhiskers(model, est.pose, K);
            
            //Mat cannyTest;
            //canny.copyTo(cannyTest);
            
            // Sample along the model edges and find the edges that intersect each whisker
            Mat targetPoints = Mat(2, 0, CV_32S);
            Mat whiskerModel = Mat(4, 0, CV_32FC1);
            for (int w = 0; w < whiskers.size(); w++) {
                Point closestEdge = whiskers[w].closestEdgePoint(edges);
                if (closestEdge == Point(-1,-1)) continue;
                hconcat(whiskerModel, whiskers[w].modelCentre, whiskerModel);
                hconcat(targetPoints, Mat(closestEdge), targetPoints);
                
                //TRACE: Display the whiskers
                //circle(cannyTest, closestEdge, 3, Scalar(120));
                //circle(cannyTest, whiskers[w].centre, 3, Scalar(255));
                //line(cannyTest, closestEdge, whiskers[w].centre, Scalar(150));
            }
            //imshow("CannyTest", cannyTest);
            
            targetPoints.convertTo(targetPoints, CV_32FC1);
            
            // Use least squares to match the sampled edges to each other
            est = lsq::poseEstimateLM(est.pose, whiskerModel, targetPoints.t(), K, 2);
            
            
            double improvement = (error - est.error)/error;
            error = est.error;
            
            // Stop trying if you reduce the error by < 1% (excl. the first one)
            if (improvement < 0.01 && iterations > 1) break;
            
            iterations++;
            //waitKey(0);
        }
        cout << "Iterations = " << iterations << endl;
        
        // Stop timer and show time
        auto stop = chrono::system_clock::now();
        chrono::duration<double> frameTime = stop-start;
        double time = frameTime.count()*1000.0;
        cout << "Frame Time = " << time << " ms" << endl << endl;
        times.push_back(time);
        if (time > longestTime) longestTime = time;
        
        // Draw the shape on the image
        model->draw(frame, est.pose, K, model->colour);
        imshow("Frame", frame);
        
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
