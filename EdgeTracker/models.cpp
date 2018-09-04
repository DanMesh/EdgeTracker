//
//  models.cpp
//  GeoHash
//
//  Created by Daniel Mesham on 16/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "lsq.hpp"
#include "models.hpp"


// * * * * * * * * * * * * * * *
//      Model
// * * * * * * * * * * * * * * *

Mat Model::pointsToMat() {
    Mat ret = Mat(4, int(vertices.size()), CV_32FC1);
    for (int i = 0; i < vertices.size(); i++) {
        Point3f p = vertices[i];
        ret.at<float>(0, i) = p.x;
        ret.at<float>(1, i) = p.y;
        ret.at<float>(2, i) = p.z;
        ret.at<float>(3, i) = 1;
    }
    return ret * 1;
}


// * * * * * * * * * * * * * * *
//      Box
// * * * * * * * * * * * * * * *

const vector<vector<float>> Box::xAngleLimits = {
    {1.0*CV_PI, 1.5*CV_PI}, {1.0*CV_PI, 1.5*CV_PI},
    {0.5*CV_PI, 1.0*CV_PI}, {0.5*CV_PI, 1.0*CV_PI},
    {1.5*CV_PI, 2.0*CV_PI}, {1.5*CV_PI, 2.0*CV_PI},
    {0.0*CV_PI, 0.5*CV_PI}, {0.0*CV_PI, 0.5*CV_PI}
};
const vector<vector<float>> Box::yAngleLimits = {
    {0.0*CV_PI, 0.5*CV_PI}, {1.5*CV_PI, 2.0*CV_PI},
    {1.5*CV_PI, 2.0*CV_PI}, {0.0*CV_PI, 0.5*CV_PI},
    {0.0*CV_PI, 0.5*CV_PI}, {1.5*CV_PI, 2.0*CV_PI},
    {1.5*CV_PI, 2.0*CV_PI}, {0.0*CV_PI, 0.5*CV_PI}
};

const vector<vector<int>> Box::faces = {
    {0,1,2,3}, {0,1,5,4}, {0,3,7,4},
    {4,5,6,7}, {1,2,6,5}, {2,3,7,6}
};

bool Box::vertexIsVisible(int vertexID, float xAngle, float yAngle) {
    while (yAngle < 0)          yAngle += 2*CV_PI;
    while (yAngle >= 2*CV_PI)   yAngle -= 2*CV_PI;
    if (yAngle > 0.5*CV_PI && yAngle < 1.5*CV_PI) {
        yAngle = CV_PI - yAngle;            // Bring into band around 0 radians
        if (yAngle < 0) yAngle += 2*CV_PI;  // Make angle positive again
        xAngle += CV_PI;                    // Add PI to the xAngle
    }
    
    while (xAngle < 0)          xAngle += 2*CV_PI;
    while (xAngle >= 2*CV_PI)   xAngle -= 2*CV_PI;
    
    vector<float> xLim = xAngleLimits[vertexID];
    vector<float> yLim = yAngleLimits[vertexID];
    if (xAngle >= xLim[0] && xAngle <= xLim[1]) {
        if (yAngle >= yLim[0] && yAngle <= yLim[1]) {
            return false;
        }
    }
    return true;
}

vector<bool> Box::visibilityMask(float xAngle, float yAngle) {
    vector<bool> mask;
    for (int i = 0; i < 8; i++) {
        mask.push_back(vertexIsVisible(i, xAngle, yAngle));
    }
    return mask;
}

void Box::createPoints(float w, float h, float d) {
    w = w/2;
    h = h/2;
    d = d/2;
    Point3f p0 = Point3f(-w, -h, -d);
    Point3f p1 = Point3f( w, -h, -d);
    Point3f p2 = Point3f( w,  h, -d);
    Point3f p3 = Point3f(-w,  h, -d);
    Point3f p4 = Point3f(-w, -h,  d);
    Point3f p5 = Point3f( w, -h,  d);
    Point3f p6 = Point3f( w,  h,  d);
    Point3f p7 = Point3f(-w,  h,  d);
    vertices = {p0, p1, p2, p3, p4, p5, p6, p7};
}

void Box::draw(Mat img, Vec6f pose, Mat K, Scalar colour, int rotOrder) {
    Mat proj = lsq::projection(pose, pointsToMat(), K, rotOrder);
    
    // Create a list of points
    vector<Point> points;
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points.push_back(Point(col.at<float>(0), col.at<float>(1)));
    }
    
    // Draw the points according to the edge list
    for (int i = 0; i < faces.size(); i++) {
        vector<int> face = faces[i];
        
        if (!vertexIsVisible(face[0], pose[3], pose[4])) continue; // Don't show invisible vertices
        if (!vertexIsVisible(face[1], pose[3], pose[4])) continue;
        if (!vertexIsVisible(face[2], pose[3], pose[4])) continue;
        if (!vertexIsVisible(face[3], pose[3], pose[4])) continue;
        
        Point pts[1][4] = {
            {points[ face[0] ], points[ face[1] ], points[ face[2] ], points[ face[3] ]}
        };
        const Point* ppt[1] = {pts[0]};
        int npt[] = {4};
        fillPoly(img, ppt, npt, 1, colour*(1 - i*0.1));
    }
}


// * * * * * * * * * * * * * * *
//      Rectangle
// * * * * * * * * * * * * * * *

bool Rectangle::vertexIsVisible(int vertexID, float xAngle, float yAngle) {
    return true;
}

vector<bool> Rectangle::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true};
}

void Rectangle::createPoints(float w, float h) {
    w = w/2;
    h = h/2;
    Point3f p0 = Point3f(-w, -h, 0);
    Point3f p1 = Point3f( w, -h, 0);
    Point3f p2 = Point3f( w,  h, 0);
    Point3f p3 = Point3f(-w,  h, 0);
    vertices = {p0, p1, p2, p3};
}

void Rectangle::draw(Mat img, Vec6f pose, Mat K, Scalar colour, int rotOrder) {
    Mat proj = lsq::projection(pose, pointsToMat(), K, rotOrder);
    
    // Create a list of points
    Point points[1][4];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    const Point* ppt[1] = {points[0]};
    int npt[] = {4};
    fillPoly(img, ppt, npt, 1, colour);
}


// * * * * * * * * * * * * * * *
//      Dog
// * * * * * * * * * * * * * * *

Dog::Dog(Scalar colourIn) {
    colour = colourIn;
    vertices = {
        Point3f(0, 0, 0), Point3f(0, -50, 0), Point3f(-10, -60, 0), Point3f(-30, -50, 0), Point3f(-38, -70, 0),
        Point3f(-10, -90, 0), Point3f(-10, -100, 0), Point3f(20, -70, 0), Point3f(80, -70, 0), Point3f(90, -90, 0),
        Point3f(90, 0, 0), Point3f(70, 0, 0), Point3f(70, -30, 0), Point3f(20, -30, 0), Point3f(20, 0, 0)
    };
    edgeBasisList = {
        {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6}, {6,7}, {7,8}, {8,9}, {9,10}, {10,11}, {11,12}, {12,13}, {13,14}, {14,15}, {15,0},
        {1,0}, {2,1}, {3,2}, {4,3}, {5,4}, {6,5}, {7,6}, {8,7}, {9,8}, {10,9}, {11,10}, {12,11}, {13,12}, {14,13}, {15,14}, {0,15}
    };
};

vector<bool> Dog::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true};
}

void Dog::draw(Mat img, Vec6f pose, Mat K, Scalar colour, int rotOrder) {
    Mat proj = lsq::projection(pose, pointsToMat(), K, rotOrder);
    
    // Create a list of points
    Point points[1][15];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    const Point* ppt[1] = {points[0]};
    int npt[] = {15};
    fillPoly(img, ppt, npt, 1, colour);
}


// * * * * * * * * * * * * * * *
//      Arrow
// * * * * * * * * * * * * * * *

Arrow::Arrow(Scalar colourIn) {
    colour = colourIn;
    vertices = {
        Point3f(-75, -15, 0), Point3f(0, -15, 0), Point3f(0, -37.5, 0), Point3f(60, 0, 0),
        Point3f(0, 37.5, 0), Point3f(0, 15, 0), Point3f(-75, 15, 0)
    };
    edgeBasisList = {
        {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6}, {6,0},
        {1,0}, {2,1}, {3,2}, {4,3}, {5,4}, {6,5}, {0,6}
    };
};

vector<bool> Arrow::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true, true, true, true};
}

void Arrow::draw(Mat img, Vec6f pose, Mat K, Scalar colour, int rotOrder) {
    Mat proj = lsq::projection(pose, pointsToMat(), K, rotOrder);
    
    // Create a list of points
    Point points[1][7];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    const Point* ppt[1] = {points[0]};
    int npt[] = {7};
    fillPoly(img, ppt, npt, 1, colour);
}
