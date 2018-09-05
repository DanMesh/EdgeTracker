//
//  asm.cpp
//  EdgeTracker
//
//  Created by Daniel Mesham on 03/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "asm.hpp"


// * * * * * * * * * * * * * * *
//      Whisker
// * * * * * * * * * * * * * * *

Point Whisker::closestEdgePoint(Mat edges, int maxDist) {
    // Edges: the result of using findNonZero() on a Canny edge datected image
    //          -> each row is an x,y coordinate
    
    double maxDist2 = maxDist*maxDist;  // Use the square of the distance for efficiency
    Point bestPt = Point(-1, -1);       // Assume no points close enough
    
    for (int i = 0; i < edges.rows; i++) {
        Point pt = Point(edges.at<int>(i,0), edges.at<int>(i,1));
        Point delta = pt - centre;
        
        // If the edge point is too far away, skip it
        double dist2 = delta.dot(delta);
        if (dist2 > maxDist2) continue;
        
        // Cross the normal and 'delta'
        double cross = normal.cross(delta);
        
        // If cross product is small, make this the best point
        if (abs(cross) < CROSS_EPS) {
            bestPt = pt;
            maxDist2 = dist2;
        }
        //return pt;
    }
    return bestPt;
}


// * * * * * * * * * * * * * * *
//      ASM
// * * * * * * * * * * * * * * *

Point ASM::getCentroid(InputArray img) {
    // Calculates the centroid of a binarised image
    // From: http://answers.opencv.org/question/94879/get-mean-coordinates-of-active-pixels/
    Point Coord;
    Moments mm = moments(img, false);
    double moment10 = mm.m10;
    double moment01 = mm.m01;
    double moment00 = mm.m00;
    Coord.x = int(moment10 / moment00);
    Coord.y = int(moment01 / moment00);
    return Coord;
}

double ASM::getArea(InputArray img) {
    // Calculates the area of a contour around a binarised image
    Point Coord;
    Moments mm = moments(img, false);
    return mm.m00;
}

vector<Whisker> ASM::projectToWhiskers(Model * model, Vec6f pose, Mat K) {
    
    vector<Whisker> whiskers = {};
    
    Mat modelMat = model->pointsToMat();
    
    Mat proj = lsq::projection(pose, modelMat, K);
    vector<bool> vis = model->visibilityMask(pose[3], pose[4]);
    
    vector<vector<int>> edges = model->getEdgeBasisList();
    for (int i = 0; i < edges.size(); i++) {
        if (!vis[edges[i][0]] || !vis[edges[i][1]]) continue;
        Mat p0_mat = proj.col(edges[i][0]);
        Mat p1_mat = proj.col(edges[i][1]);
        Point p0 = Point(p0_mat.at<float>(0), p0_mat.at<float>(1));
        Point p1 = Point(p1_mat.at<float>(0), p1_mat.at<float>(1));
        
        // Calculate the edge vector and length
        Point edge = p1 - p0;
        double length = sqrt(edge.dot(edge));
        
        // Calculate a unit normal and the side length
        Point2f normal = Point2f(edge.y, -edge.x);
        normal /= length;
        
        // Divide up the edge
        int numWhiskers = MAX(1, ceil(length/WHISKER_SPACING - 0.5));
        double offset = 0.5 * (length - (numWhiskers-1) * WHISKER_SPACING);
        
        // Do the same for the model points
        Mat m0_mat = modelMat.col(edges[i][0]);
        Mat m1_mat = modelMat.col(edges[i][1]);
        
        for (int w = 0; w < numWhiskers; w++) {
            double distance = offset + w * WHISKER_SPACING;
            Point centre = p0 + edge * (distance/length);
            Mat centre_mat = m0_mat + (m1_mat - m0_mat) * (distance/length);
            whiskers.push_back(Whisker(centre, normal, centre_mat));
        }
    }
    
    return whiskers;
}

