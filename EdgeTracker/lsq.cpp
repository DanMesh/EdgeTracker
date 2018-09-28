//
//  lsq.cpp
//  LiveTracker
//
//  Created by Daniel Mesham on 07/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "lsq.hpp"

estimate lsq::poseEstimateLM(Vec6f pose1, Mat model, Mat target, Mat K, int maxIter) {
    // pose1: imitial pose parameters
    // model: model points in full homogeneous coords
    // target: image points, in 2D coords
    // K: intrinsic matrix
    // maxIter: max no of iterations, default if 0
    // rotOrder: order of rotations, default XYZ
    
    if (maxIter == 0) maxIter = MAX_ITERATIONS;
    
    Mat y = lsq::projection(pose1, model, K);
    float E = lsq::projectionError(target, y);
    
    int iterations = 0;
    while (E > ERROR_THRESHOLD && iterations < maxIter) {
        Mat J = lsq::jacobian(pose1, model, K);
        Mat eps;
        subtract(y.rowRange(0, 2).t(), target, eps);
        eps = lsq::pointsAsCol(eps.t());
        for (int i = 0; i < eps.rows; i++) {
            if (eps.at<float>(i) > 20) eps.at<float>(i) = 20;
        }
        Mat Jp = J.t() * J;
        Jp = -Jp.inv() * J.t();
        Mat del = Jp * eps;
        
        Vec6f pose2 = pose1;
        for (int i = 0; i < 6; i++) {
            pose2[i] += del.at<float>(i);
        }
        
        y = lsq::projection(pose2, model, K);
        E = lsq::projectionError(target, y);
        
        pose1 = pose2;
        iterations++;
    }
    
    return estimate(pose1, E, iterations);
}

/*
 Method for optimising point-distance errors as well as colour errors.
 */
estimate lsq::poseEstimateLM(Vec6f pose1, Mat model, Mat target, Mat K, Mat imgHue, Scalar colour, Mat colourPoints, float alpha, int maxIter) {
    // pose1: imitial pose parameters
    // model: model points in full homogeneous coords
    // target: image points, in 2D coords
    // K: intrinsic matrix
    // imgHue: the Hue channel of the HSV image
    // colour: the expected colour of certain points
    // colourPoints: the points that should have the colour provided
    // alpha: the weighting between distance errors (0) and colour errors (1)
    // maxIter: max no of iterations, default if 0
    
    if (maxIter == 0) maxIter = MAX_ITERATIONS;
    
    Mat bgr(1 ,1 , CV_8UC3, colour);
    Mat3b hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    int hue = hsv.at<int>(0,0);
    
    Mat y = lsq::projection(pose1, model, K);
    Mat yC = lsq::projection(pose1, colourPoints, K);
    float E = (1-alpha)*lsq::projectionError(target, y) + (alpha)*pow(lsq::colourError(imgHue, yC, hue), 2);
    
    int iterations = 0;
    while (E > ERROR_THRESHOLD && iterations < maxIter) {
        Mat J = lsq::jacobian(pose1, model, K) * (1-alpha);cout << J << endl << endl;
        vconcat(J, alpha*jacobianColour(pose1, colourPoints, K, imgHue), J);
        Mat eps;
        subtract(y.rowRange(0, 2).t(), target, eps);
        eps = lsq::pointsAsCol(eps.t());
        for (int i = 0; i < eps.rows; i++) {
            if (eps.at<float>(i) > 20) eps.at<float>(i) = 20;
        }
        vconcat(eps, coloursAtPoints(imgHue, yC) - hue, eps);
        Mat Jp = J.t() * J;
        Jp = -Jp.inv() * J.t();
        Mat del = Jp * eps;
        
        Vec6f pose2 = pose1;
        for (int i = 0; i < 6; i++) {
            pose2[i] += del.at<float>(i);
        }
        
        y = lsq::projection(pose2, model, K);
        yC = lsq::projection(pose2, colourPoints, K);
        E = (1-alpha)*lsq::projectionError(target, y) + (alpha)*pow(lsq::colourError(imgHue, yC, hue), 2);
        
        pose1 = pose2;
        iterations++;
    }
    
    return estimate(pose1, E, iterations);
}


Mat lsq::translation(float x, float y, float z) {
    // Translate by the given x, y and z values
    float tmp[] = {x, y, z};
    return Mat(3, 1, CV_32FC1, tmp) * 1;
}

Mat lsq::rotation(float x, float y, float z) {
    // Rotate about the x, y then z axes with the given angles in radians
    float rotX[3][3] = {
        { 1,       0,       0 },
        { 0,  cos(x), -sin(x) },
        { 0,  sin(x),  cos(x) }
    };
    float rotY[3][3] = {
        {  cos(y),   0,  sin(y) },
        {       0,   1,       0 },
        { -sin(y),   0,  cos(y) }
    };
    float rotZ[3][3] = {
        {  cos(z), -sin(z),  0 },
        {  sin(z),  cos(z),  0 },
        {       0,       0,  1 }
    };
    
    Mat rX = Mat(3, 3, CV_32FC1, rotX);
    Mat rY = Mat(3, 3, CV_32FC1, rotY);
    Mat rZ = Mat(3, 3, CV_32FC1, rotZ);
    
    return rZ * rY * rX;
}

Mat lsq::projection(Vec6f pose, Mat model, Mat K) {
    Mat P;
    hconcat( rotation(pose[3], pose[4], pose[5]) , translation(pose[0], pose[1], pose[2]) , P);
    Mat y = (K * P) * model;
    
    Mat z = y.row(2);
    Mat norm;
    vconcat(z, z, norm);
    vconcat(norm, z, norm);
    divide(y, norm, y);
    return y;
}

float lsq::projectionError(Mat target, Mat proj) {
    transpose(proj.rowRange(0, 2), proj);
    
    Mat e;
    subtract(target, proj, e);
    multiply(e, e, e);
    reduce(e, e, 1, CV_REDUCE_SUM, CV_32FC1);
    sqrt(e, e);
    Mat eT;
    transpose(e, eT);
    e = eT * e;
    return e.at<float>(0);
}

float lsq::colourError(Mat imgH, Mat points, int hue) {
    // Returns the sum of the squared errors in the Hue (H) of the image
    // at the given points
    // imgH: the Hue (H) channel of the image
    // points: the points at which to sample the colour
    // hue: the expected Hue value
    return 0;
}

Mat lsq::coloursAtPoints(Mat imgHue, Mat points) {
    Mat ret = Mat(points.cols, 1, CV_32FC1);
    for (int i = 0; i < points.cols; i++) {
        ret.at<float>(i, 0) = imgHue.at<u_char>(points.at<float>(0, i), points.at<float>(1, i));
    }
    return ret;
}

Mat lsq::pointsAsCol(Mat points) {
    // Converts a matrix of points (each a column vector in homogeneous coordinates) into a single column of coordinates in standard coordinates (i.e. the last coordinate is removed)
    points = points.rowRange(0, 2);
    points = points.t();
    points = points.reshape(0, 1).t();
    return points;
}

Mat lsq::jacobian(Vec6f pose, Mat model, Mat K) {
    // Calculates the Jacobian for the given pose of model x
    Mat J = Mat(2*model.cols, 0, CV_32FC1);
    
    float dt = 1;
    float dr = CV_PI/180;
    vector<float> delta = {dt, dt, dt, dr, dr, dr};
    
    for (int i = 0; i < 6; i++) {
        Vec6f p1 = pose;
        p1[i] += delta[i];
        Vec6f p2 = pose;
        p2[i] -= delta[i];
        Mat j = (projection(p1, model, K) - projection(p2, model, K))/delta[i];
        hconcat(J, pointsAsCol(j), J);
    }
    
    return J;
}

Mat lsq::jacobianColour(Vec6f pose, Mat points, Mat K, Mat imgHue) {
    // Calculates the Jacobian of the Hues at the given points for the given pose of model x.
    // Each column of 'points' is a point
    Mat J = Mat(points.cols, 0, CV_32FC1);
    
    float dt = 1;
    float dr = CV_PI/180;
    vector<float> delta = {dt, dt, dt, dr, dr, dr};
    
    for (int i = 0; i < 6; i++) {
        Vec6f p1 = pose;
        p1[i] += delta[i];
        Vec6f p2 = pose;
        p2[i] -= delta[i];
        
        Mat points1 = projection(p1, points, K);
        Mat points2 = projection(p2, points, K);
        
        Mat dC = Mat(points.cols, 1, CV_32FC1); // The colour gradients of each point
        
        // For each point calculate the colour gradient
        for (int j = 0; j < points.cols; j++) {
            int c1 = imgHue.at<u_char>(points1.at<float>(0, j), points1.at<float>(1, j));
            int c2 = imgHue.at<u_char>(points2.at<float>(0, j), points2.at<float>(1, j));
            dC.at<float>(j, 0) = (c1 - c2)/delta[i];
        }
        hconcat(J, dC, J);
    }
    
    return J;
}

Vec6f estimate::standardisePose(Vec6f pose) {
    // Ensures all angles are in (-PI,PI]
    for (int i = 3; i < 6; i++) {
        while (pose[i] > CV_PI)     pose[i] -= CV_2PI;
        while (pose[i] <= -CV_PI)   pose[i] += CV_2PI;
    }
    return pose;
}
