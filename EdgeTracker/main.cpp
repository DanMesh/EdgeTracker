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
#include "models.hpp"
#include "orange.hpp"

#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;


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

// The initial Arrow endpoints in Trio_Vid.avi
static float arrowInit[2][7] = {
    {395, 640, 584, 839, 818, 737, 486},
    {516, 351, 306, 277, 470, 415, 607}
};
static Mat arrowTargetInit = Mat(2,7, CV_32FC1, arrowInit);

static string dataFolder = "../../../../../data/";



// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Main Method
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *


int main(int argc, const char * argv[]) {
    
    // * * * * * * * * * * * * * * * * *
    //   MODEL CREATION
    // * * * * * * * * * * * * * * * * *
    
    //Model * model = new Rectangle(60, 80, Scalar(20, 65, 165));
    //Model * model = new Dog(Scalar(19, 89, 64));
    Model * model = new Arrow(Scalar(108, 79, 28));
    
    Mat modelMat = model->pointsToMat();
    vector<Point3f> modelPoints = model->getVertices();
    
    // * * * * * * * * * * * * * * * * *
    //   OPEN THE FIRST FRAME
    // * * * * * * * * * * * * * * * * *
    
    Mat frame;
    String filename = "Trio_Vid.avi";
    VideoCapture cap(dataFolder + filename);
    if(!cap.isOpened()) return -1;
    
    cap >> frame;
    imshow(filename, frame);
   
    
    // Build an initial pose estimate based on previously measured points
    Vec6f pose = {0, 0, 300, 0, 0, 0};
    estimate est = lsq::poseEstimateLM(pose, modelMat, arrowTargetInit.t(), K);
    pose = est.pose;
    cout << pose << endl;
    cout << est.iterations << endl;
    cout << est.error << endl << endl;
    
    
    
    // * * * * * * * * * * * * * * * * *
    //   CAMERA LOOP
    // * * * * * * * * * * * * * * * * *
    
    while (!frame.empty()) {
        
        imshow("Frame", frame);
        
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
            vector<Whisker> whiskers = ASM::projectToWhiskers(model, pose, K);
            
            // Sample along the model edges and find the edges that intersect each whisker
            Mat targetPoints = Mat(whiskers[0].closestEdgePoint(edges));
            Mat whiskerModel = whiskers[0].modelCentre;
            for (int w = 1; w < whiskers.size(); w++) {
                Point closestEdge = whiskers[w].closestEdgePoint(edges);
                if (closestEdge == Point(-1,-1)) continue;  // Don't use points with nothing close enough
                hconcat(whiskerModel, whiskers[w].modelCentre, whiskerModel);
                hconcat(targetPoints, Mat(closestEdge), targetPoints);
            }
            
            targetPoints.convertTo(targetPoints, CV_32FC1);
            
            // Use least squares to match the sampled edges to each other
            est = lsq::poseEstimateLM(pose, whiskerModel, targetPoints.t(), K, 40);
            
            iterations++;
            double improvement = (error - est.error)/error;
            error = est.error;
            if (improvement < 0.01) break;  // Stop trying if you reduce the error by < 1%
        }
        cout << "Did that " << iterations << " times" << endl;
        cout << est.pose << endl << est.error << endl << endl;
        
        Mat canny2;
        canny.copyTo(canny2);
        model->draw(canny2, est.pose, K, Scalar(255, 255, 255), lsq::ROT_XYZ);
        imshow("Canny", canny);
        imshow("Canny + Matched", canny2);
        
        // Get next frame
        cap.grab();
        cap >> frame;
        
        if (waitKey(0) == 'q') break;
    }
    
    return 0;
}
