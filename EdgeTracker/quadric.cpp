//
//  quadric.cpp
//  EdgeTracker
//
//  Created by Daniel Mesham on 11/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "quadric.hpp"


// * * * * * * * * * * * * * * *
//      Plane
// * * * * * * * * * * * * * * *

Plane::Plane(Point3f normal, Point3f point) {
    pi = Mat(normal);
    Mat pt = Mat(point);
    Mat w = -pi.t()*pt;
    vconcat(pi, w, pi);
};


// * * * * * * * * * * * * * * *
//      Conic
// * * * * * * * * * * * * * * *
// See Hartley & Zisserman, p. 30

vector<bool> Conic::pointsOnModel(Mat points) {
    // points: 2D column vector coordinates in homogeneous coordinates
    // Returns a vector of booleans indicating true if the corresponding point is on the model
    float THRESHOLD = 0.5;
    
    Mat d = points.t() * C * points;
    cout << d << endl;
    vector<bool> ret(points.cols);
    for (int p = 0; p < points.cols; p++) {
        if (abs(d.at<float>(p,p)) < THRESHOLD) ret[p] = true;
    }
    return ret;
}

Conic Conic::circle(Point2f centre, float radius, Scalar colour) {
    // Creates a circle with the given centre and radius
    float x2 = 1;
    float y2 = 1;
    float x = -2*centre.x;
    float y = -2*centre.y;
    float c = pow(centre.x, 2) + pow(centre.y, 2) - pow(radius, 2);
    float C[3][3] = {
        { x2,   0, x/2},
        {  0,  y2, y/2},
        {x/2, y/2,   c}
    };
    return Conic(Mat(3, 3, CV_32F, C) * 1, colour);
}

bool Conic::isEllipse() {
    // From: http://code.i-harness.com/en/q/b0367a
    // Compute SVD
    Mat w, u, vt;
    SVD::compute(C, w, u, vt);
    
    // w is the matrix of singular values
    // Find non zero singular values.
    
    // Use a small threshold to account for numeric errors
    Mat nonZeroSingularValues = w > 0.0001;
    
    // Count the number of non zero
    int rank = countNonZero(nonZeroSingularValues);
    
    // Is an ellipse if rank == 3 (Cipolla et al)
    return rank == 3;
}


// * * * * * * * * * * * * * * *
//      Quadric
// * * * * * * * * * * * * * * *

void Quadric::makeBounds(vector<Plane> bounds) {
    // Creates the "pi" matrix from the 2 bounding planes
    if (bounds.size() != 2) return;
    hconcat(bounds[0].pi, bounds[1].pi, pi);
}

vector<bool> Quadric::pointsBetweenBounds(Mat points) {
    // points: 3D column vector coordinates in homogeneous coordinates
    // Returns a vector of booleans indicating true if the corresponding point is in the bounds
    vector<bool> ret(points.cols);
    if (pi.empty()) {
        for (int p = 0; p < points.cols; p++) ret[p] = true;
    }
    else {
        Mat check = pi.t() * points;
        for (int i = 0; i < check.cols; i++) {
            ret[i] = check.at<float>(0, i) >= 0 && check.at<float>(1, i) >= 0;
        }
    }
    return ret;
}

vector<bool> Quadric::pointsOnModel(Mat points) {
    // points: 3D column vector coordinates in homogeneous coordinates
    // Returns a vector of booleans indicating true if the corresponding point is on the model
    float THRESHOLD = 0.5;
    
    Mat d = points.t() * Q * points;
    cout << d << endl;
    vector<bool> inBounds = pointsBetweenBounds(points);
    vector<bool> ret(points.cols);
    for (int p = 0; p < points.cols; p++) {
        if (abs(d.at<float>(p,p)) < THRESHOLD) ret[p] = inBounds[p];
    }
    return ret;
}

Conic Quadric::imageFromView() {
    // Returns the conic visible from a normalised  projective camera
    // See Cipolla et al, p. 3
    Mat A = Q.colRange(0, 3);
    A = A.rowRange(0, 3); cout << A << endl;
    Mat b = Q.col(3);
    float c = b.at<float>(3);
    b = b.rowRange(0, 3); cout << b << endl;
    Mat C = c*A - b*b.t(); cout << c << endl;
    return Conic(C * 1, colour);
}



