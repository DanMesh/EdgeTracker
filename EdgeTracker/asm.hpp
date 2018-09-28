//
//  asm.hpp
//  EdgeTracker
//
//  Created by Daniel Mesham on 03/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef asm_hpp
#define asm_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "models.hpp"

using namespace std;
using namespace cv;


class Whisker {
public:
    Whisker(Point c_in, Point2f n_in, Mat mp_in) : centre(c_in), normal(n_in), modelCentre(mp_in) {}
    Point centre;
    Point2f normal;
    Mat modelCentre;
    Point closestEdgePoint(Mat edges, int maxDist = MAX_DIST);
    Point closestEdgePoint2(Mat canny, int maxDist = MAX_DIST);
    Point closestEdgePoint2(Mat canny[3], int maxDist = MAX_DIST);
private:
    static const int MAX_DIST = 45;
    static constexpr double CROSS_EPS = 1;
};


class ASM {
public:
    static Point getCentroid(InputArray img);
    static double getArea(InputArray img);
    static vector<Whisker> projectToWhiskers(Model * model, Vec6f pose, Mat K);
private:
    static constexpr double WHISKER_SPACING = 20;
};

#endif /* asm_hpp */
