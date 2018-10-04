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

vector<bool> Box::faceVisibilityMask(Vec6f pose) {
    vector<bool> ret(6);
    
    // Find normals after rotation
    float normalArr[3][6] = {
        { 0,  0, -1,  0,  1,  0},
        { 0, -1,  0,  0,  0,  1},
        {-1,  0,  0,  1,  0,  0}
    };
    Mat normals = Mat(3, 6, CV_32F, normalArr);
    Mat R = lsq::rotation(pose[3], pose[4], pose[5]);
    normals = R * normals;
    
    // Find translation (of centre point)
    float translation[3] = {pose[0], pose[1], pose[2]};
    Mat t = Mat(3, 1, CV_32F, translation);
    
    for (int f = 0; f < normals.cols; f++) {
        Mat n = normals.col(f);
        float magN = normMag.at<float>(f);
        
        // Find the translation of the centre of the face
        Mat t2 = t + n;
        t2 = t2.mul(t2);
        reduce(t2, t2, 1, CV_REDUCE_SUM, CV_32F);
        float magT = sqrt(t2.at<float>(0));
        
        double dot = t.dot(n);
        
        // Cosine of the angle between the normal and view (translation) must be +ive
        ret[f] = -dot / (magT * magN) > 0.03;
    }
    
    return ret;
}

vector<bool> Box::visibilityMask(float xAngle, float yAngle) {
    vector<bool> mask;
    for (int i = 0; i < 8; i++) {
        mask.push_back(vertexIsVisible(i, xAngle, yAngle));
    }
    return mask;
}

vector<bool> Box::visibilityMask(Vec6f pose) {
    vector<bool> mask(8);
    vector<bool> maskFaces = faceVisibilityMask(pose);
    for (int f = 0; f < 6; f++) {
        if (maskFaces[f]) {
            for (int v = 0; v < 4; v++) {
                mask[ faces[f][v] ] = true;
            }
        }
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
    
    float normalArr[3][6] = {
        { 0,  0, -w,  0,  w,  0},
        { 0, -h,  0,  0,  0,  h},
        {-d,  0,  0,  d,  0,  0}
    };
    normals = Mat(3, 6, CV_32F, normalArr) * 1;
    float normMagArr[6] = {d, h, w, d, w, h};
    normMag = Mat(1, 6, CV_32F, normMagArr) * 1;
}

void Box::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    vector<Point> points;
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points.push_back(Point(col.at<float>(0), col.at<float>(1)));
    }
    
    //vector<bool> faceVis = faceVisibilityMask(pose);
    
    // Draw the points according to the edge list
    for (int i = 0; i < faces.size(); i++) {
        //if (!faceVis[i]) continue;  // Don't show invisible vertices
        
        vector<int> face = faces[i];
        
        Point pts[1][4] = {
            {points[ face[0] ], points[ face[1] ], points[ face[2] ], points[ face[3] ]}
        };
        const Point* ppt[1] = {pts[0]};
        int npt[] = {4};
        
        if (lines) polylines(img, ppt, npt, 1, true, colour);
        else fillPoly(img, ppt, npt, 1, colour*(1 - i*0.1));
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

void Rectangle::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    Point points[1][4];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    if (!lines) {
        const Point* ppt[1] = {points[0]};
        int npt[] = {4};
        fillPoly(img, ppt, npt, 1, colour);
    }
    else {
        for (int i = 0; i < edgeBasisList.size()/2; i++) {
            vector<int> edge = edgeBasisList[i];
            line(img, points[0][edge[0]], points[0][edge[1]], colour, 1);
        }
    }
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
        {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6}, {6,7}, {7,8}, {8,9}, {9,10}, {10,11}, {11,12}, {12,13}, {13,14}, {14,0},
        {1,0}, {2,1}, {3,2}, {4,3}, {5,4}, {6,5}, {7,6}, {8,7}, {9,8}, {10,9}, {11,10}, {12,11}, {13,12}, {14,13}, {0,14}
    };
    is3D = false;
};

vector<bool> Dog::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true};
}

void Dog::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    Point points[1][15];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    if (!lines) {
        const Point* ppt[1] = {points[0]};
        int npt[] = {15};
        fillPoly(img, ppt, npt, 1, colour);
    }
    else {
        for (int i = 0; i < edgeBasisList.size()/2; i++) {
            vector<int> edge = edgeBasisList[i];
            line(img, points[0][edge[0]], points[0][edge[1]], colour, 1);
        }
    }
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
    is3D = false;
};

vector<bool> Arrow::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true, true, true, true};
}

void Arrow::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    Point points[1][7];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    if (!lines) {
        const Point* ppt[1] = {points[0]};
        int npt[] = {7};
        fillPoly(img, ppt, npt, 1, colour);
    }
    else {
        for (int i = 0; i < edgeBasisList.size()/2; i++) {
            vector<int> edge = edgeBasisList[i];
            line(img, points[0][edge[0]], points[0][edge[1]], colour, 1);
        }
    }
}

// * * * * * * * * * * * * * * *
//      Triangle
// * * * * * * * * * * * * * * *

Triangle::Triangle(Scalar colourIn) {
    colour = colourIn;
    vertices = {
        Point3f(0, 0, 0), Point3f(65, 65, 0), Point3f(-65, 65, 0)
    };
    edgeBasisList = {
        {0,1}, {1,2}, {2,0},
        {1,0}, {2,1}, {0,2}
    };
    is3D = false;
};

vector<bool> Triangle::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true};
}

void Triangle::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    Point points[1][3];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    if (!lines) {
        const Point* ppt[1] = {points[0]};
        int npt[] = {3};
        fillPoly(img, ppt, npt, 1, colour);
    }
    else {
        for (int i = 0; i < edgeBasisList.size()/2; i++) {
            vector<int> edge = edgeBasisList[i];
            line(img, points[0][edge[0]], points[0][edge[1]], colour, 1);
        }
    }
}

// * * * * * * * * * * * * * * *
//      Diamond
// * * * * * * * * * * * * * * *

Diamond::Diamond(Scalar colourIn) {
    colour = colourIn;
    vertices = {
        Point3f(0, 0, 0), Point3f(-50, -60, 0), Point3f(-30, -85, 0), Point3f(30, -85, 0),
        Point3f(50, -60, 0)
    };
    edgeBasisList = {
        {0,1}, {1,2}, {2,3}, {3,4}, {4,0},
        {1,0}, {2,1}, {3,2}, {4,3}, {0,4}
    };
    is3D = false;
};

vector<bool> Diamond::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true, true};
}

void Diamond::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    Point points[1][5];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    if (!lines) {
        const Point* ppt[1] = {points[0]};
        int npt[] = {5};
        fillPoly(img, ppt, npt, 1, colour);
    }
    else {
        for (int i = 0; i < edgeBasisList.size()/2; i++) {
            vector<int> edge = edgeBasisList[i];
            line(img, points[0][edge[0]], points[0][edge[1]], colour, 1);
        }
    }
}

// * * * * * * * * * * * * * * *
//      House
// * * * * * * * * * * * * * * *

House::House(Scalar colourIn) {
    colour = colourIn;
    vertices = {
        Point3f(0, 0, 0), Point3f(0, -45, 0), Point3f(-15, -45, 0), Point3f(40, -90, 0),
        Point3f(95, -45, 0), Point3f(80, -45, 0), Point3f(80, 0, 0)
    };
    edgeBasisList = {
        {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6}, {6,0},
        {1,0}, {2,1}, {3,2}, {4,3}, {5,4}, {6,5}, {0,6}
    };
    is3D = false;
};

vector<bool> House::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true, true, true, true};
}

void House::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    Point points[1][7];
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points[0][i] = Point(col.at<float>(0), col.at<float>(1));
    }
    
    if (!lines) {
        const Point* ppt[1] = {points[0]};
        int npt[] = {7};
        fillPoly(img, ppt, npt, 1, colour);
    }
    else {
        for (int i = 0; i < edgeBasisList.size()/2; i++) {
            vector<int> edge = edgeBasisList[i];
            line(img, points[0][edge[0]], points[0][edge[1]], colour, 1);
        }
    }
}
