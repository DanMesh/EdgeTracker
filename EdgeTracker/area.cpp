//
//  area.cpp
//  EdgeTracker
//
//  Created by Daniel Mesham on 05/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "area.hpp"


double area::areaError(Vec6f pose, Model * model, Mat img, Mat K) {
    // Count the number of pixels in the model projection
    Mat modelProj = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
    model->draw(modelProj, pose, K, false, Scalar(255));
    threshold(modelProj, modelProj, 0, 255, modelProj.type());
    int numModelPixels = countNonZero(modelProj);
    
    // Count the number of pixels in the image
    Mat imgBinary;
    cvtColor(img, imgBinary, CV_BGR2GRAY);
    int numImagePixels = countNonZero(imgBinary);
    
    // Find the overlapping region
    Mat overlap;
    bitwise_and(imgBinary, modelProj, overlap);
    int numOverlappingPixels = countNonZero(overlap);
    
    // Calculate the percentage of the UNION not included in the INTERSECTION
    // i.e. (Model XOR Image) / (Model OR Image)
    int OR = numModelPixels + numImagePixels - numOverlappingPixels;
    int XOR = OR - numOverlappingPixels;
    return 100.0 * XOR / OR;
}

Mat area::jacobian(Vec6f pose, Model * model, Mat img, Mat K) {
    // Calculates the Jacobian for the given pose of model x
    //Mat J = Mat(1, 0, CV_32FC1);
    vector<double> J = {};
    
    float dt = 0.5;
    float dr = CV_PI/360;
    vector<float> delta = {dt, dt, dt, dr, dr, dr};
    
    for (int i = 0; i < 6; i++) {
        Vec6f p1 = pose;
        p1[i] += delta[i];
        Vec6f p2 = pose;
        p2[i] -= delta[i];
       
        double j = (areaError(p1, model, img, K) - areaError(p2, model, img, K)) / (2*delta[i]);
        j = MIN(j, 100.0 / (2*delta[i]) );
        j = MAX(j, -100.0 / (2*delta[i]) );
        J.push_back( j );
        //hconcat(J, Mat(jAdd), J);
    }
        
    return Mat(J);
}



estimate area::poseEstimateArea(Vec6f pose1, Model * model, Mat img, Mat K, int maxIter) {
    // pose1: imitial pose parameters
    // model: model to be matched
    // img: segmented image mask
    // K: intrinsic matrix
    // maxIter: max no of iterations, default if 0

    if (maxIter == 0) maxIter = MAX_ITERATIONS;

    double E = areaError(pose1, model, img, K);
    cout << "E after 0 = " << E << endl;

    int iterations = 0;
    while (E > ERROR_THRESHOLD && iterations < maxIter) {
        Mat J = jacobian(pose1, model, img, K);
        Mat Jp = J.t() * J;
        Jp = -Jp.inv() * J.t();
        Mat del = Jp * E / 20.0;

        for (int i = 0; i < 6; i++) {
            pose1[i] += del.at<double>(i);
        }

        E = areaError(pose1, model, img, K);
        
        iterations++;
        cout << "E after " << iterations << " = " << E << endl;
    }

    return estimate(pose1, E, iterations);
}

double area::unexplainedArea(Vec6f pose, Model * model, Mat img, Mat K) {
    // Counts the percentage of the image pixels (in 'img') not explained by the
    // model when in the given pose.
    
    // Count the number of pixels in the model projection
    Mat modelProj = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
    model->draw(modelProj, pose, K, false);
    
    // Count the number of pixels in the image
    Mat imgBinary;
    cvtColor(img, imgBinary, CV_BGR2GRAY);
    int numImagePixels = countNonZero(imgBinary);
    
    // Find the overlapping region
    Mat overlap;
    bitwise_and(imgBinary, modelProj, overlap);
    int numOverlappingPixels = countNonZero(overlap);
    
    // Calculate the percentage of the image not in the intersection
    int numUnexplainedPixels = numImagePixels - numOverlappingPixels;
    return 100.0 * numUnexplainedPixels / numImagePixels;
}

