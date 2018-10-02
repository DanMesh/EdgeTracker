//
//  quadric.hpp
//  EdgeTracker
//
//  Created by Daniel Mesham on 11/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef quadric_hpp
#define quadric_hpp

#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;


// * * * * * * * * * * * * * * *
//      Plane
// * * * * * * * * * * * * * * *

class Plane {
public:
    Plane(Point3f normal, Point3f point);
    Mat pi;
};


// * * * * * * * * * * * * * * *
//      Conic
// * * * * * * * * * * * * * * *

class Conic {
public:
    Conic(Mat C_in, Scalar colour_in = Scalar(255,255,255)) : C(C_in), colour(colour_in) {}
    Mat C;
    Scalar colour;
    vector<bool> pointsOnModel(Mat points);
private:
    bool isEllipse();
    
public:
    static Conic circle(Point2f centre, float radius, Scalar colour = Scalar(255,255,255));
};


// * * * * * * * * * * * * * * *
//      Quadric
// * * * * * * * * * * * * * * *

class Quadric {
public:
    Quadric(Mat Q_in, vector<Plane> bounds_in = {}, Scalar colour_in = Scalar(255, 255, 255)) : Q(Q_in), colour(colour_in) {
        makeBounds(bounds_in);
    };
    Scalar colour;
    Mat Q, pi;
    vector<bool> pointsBetweenBounds(Mat points);
    vector<bool> pointsOnModel(Mat points);
    Conic imageFromView();
    //virtual void draw(Mat img, Vec6f pose, Mat K, Scalar drawColour = Scalar(255, 255, 255)) = 0;
    
private:
    void makeBounds(vector<Plane> bounds);
    
};

#endif /* quadric_hpp */
