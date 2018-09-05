//
//  models.hpp
//  GeoHash
//
//  Created by Daniel Mesham on 16/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef models_hpp
#define models_hpp

#include <opencv2/core/core.hpp>
#include <stdio.h>
#include "lsq.hpp"

using namespace std;
using namespace cv;


// * * * * * * * * * * * * * * *
//      Model
// * * * * * * * * * * * * * * *

class Model {
public:
    bool vertexIsVisible(int vertexID, float xAngle, float yAngle);
    virtual vector<bool> visibilityMask(float xAngle, float yAngle) = 0;
    vector<Point3f> getVertices() {return vertices;};
    vector<vector<int>> getEdgeBasisList() {return edgeBasisList;}
    Mat pointsToMat();
    virtual void draw(Mat img, Vec6f pose, Mat K, Scalar drawColour = Scalar(255, 255, 255)) = 0;
    Scalar colour = Scalar(255, 255, 255);
    
protected:
    vector<Point3f> vertices;
    vector<vector<int>> edgeBasisList;
    
};


// * * * * * * * * * * * * * * *
//      Box
// * * * * * * * * * * * * * * *

class Box : public Model {
public:
    Box(float width, float height, float depth) {
        createPoints(width, height, depth);
        edgeBasisList = {
            {0,1}, {1,2}, {2,3}, {3,0},
            {1,0}, {2,1}, {3,2}, {0,3},
            
            {4,5}, {5,6}, {6,7}, {7,4},
            {5,4}, {6,5}, {7,6}, {4,7},
            
            {0,4}, {1,5}, {2,6}, {3,7},
            {4,0}, {5,1}, {6,2}, {7,3}
        };
    }
    bool vertexIsVisible(int vertexID, float xAngle, float yAngle);
    vector<bool> visibilityMask(float xAngle, float yAngle);
    void draw(Mat img, Vec6f pose, Mat K, Scalar drawColour);
    
private:
    static const vector<vector<float>> xAngleLimits;
    static const vector<vector<float>> yAngleLimits;
    static const vector<vector<int>> faces;
    void createPoints(float width, float height, float depth);
};


// * * * * * * * * * * * * * * *
//      Rectangle
// * * * * * * * * * * * * * * *

class Rectangle : public Model {
public:
    Rectangle(float width, float height, Scalar colourIn) {
        createPoints(width, height);
        edgeBasisList = {
            {0,1}, {1,2}, {2,3}, {3,0},
            {1,0}, {2,1}, {3,2}, {0,3},
            
            {4,5}, {5,6}, {6,7}, {7,4},
            {5,4}, {6,5}, {7,6}, {4,7},
            
            {0,4}, {1,5}, {2,6}, {3,7},
            {4,0}, {5,1}, {6,2}, {7,3}
        };
        colour = colourIn;
    }
    bool vertexIsVisible(int vertexID, float xAngle, float yAngle);
    vector<bool> visibilityMask(float xAngle, float yAngle);
    void draw(Mat img, Vec6f pose, Mat K, Scalar drawColour);
    
private:
    void createPoints(float width, float height);
};


// * * * * * * * * * * * * * * *
//      Dog
// * * * * * * * * * * * * * * *

class Dog : public Model {
public:
    Dog(Scalar colourIn);
    bool vertexIsVisible(int vertexID, float xAngle, float yAngle) {return true;};
    vector<bool> visibilityMask(float xAngle, float yAngle);
    void draw(Mat img, Vec6f pose, Mat K, Scalar drawColour);
};


// * * * * * * * * * * * * * * *
//      Arrow
// * * * * * * * * * * * * * * *

class Arrow : public Model {
public:
    Arrow(Scalar colourIn);
    bool vertexIsVisible(int vertexID, float xAngle, float yAngle) {return true;};
    vector<bool> visibilityMask(float xAngle, float yAngle);
    void draw(Mat img, Vec6f pose, Mat K, Scalar drawColour);
};




/*
 
   z
  /             Rotation directions defined by
 .__ x          the right hand grip rule.
 |
 y
 
   4 .___. 5
    /   /|      Vertex IDs increase in x, y then z directions.
 0 .___.1. 6    i.e. front face then back face
   |   |/
 3 .___. 2
 
 */

#endif /* models_hpp */
