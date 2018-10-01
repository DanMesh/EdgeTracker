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
