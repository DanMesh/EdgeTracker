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

Point Whisker::closestEdgePoint2(Mat canny, int maxDist) {
    Point endPos = centre + Point(maxDist*normal.x, maxDist*normal.y);
    Point endNeg = centre - Point(maxDist*normal.x, maxDist*normal.y);
    if (canny.at<uchar>(centre) > 0) return centre;
    
    LineIterator liPos(canny, centre, endPos);
    LineIterator liNeg(canny, centre, endNeg);
    for (int i = 0; i < MIN(liPos.count, liNeg.count); i++, ++liPos, ++liNeg) {
        if (canny.at<uchar>(liPos.pos()) > 0) {
            // If exactly in between 2 edges, discard this whisker
            if (canny.at<uchar>(liNeg.pos()) > 0) return Point(-1,-1);
            return liPos.pos();
        }
        if (canny.at<uchar>(liNeg.pos()) > 0) return liNeg.pos();
    }
    return Point(-1,-1);
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
    
    vector<bool> vis = model->visibilityMask(pose[3], pose[4]);
    
    vector<vector<int>> edges = model->getEdgeBasisList();
    
    for (int i = 0; i < edges.size()/2; i++) {
        if (!vis[edges[i][0]] || !vis[edges[i][1]]) continue;
        
        // Get the edge endpoints
        Mat p0 = modelMat.col(edges[i][0]);
        Mat p1 = modelMat.col(edges[i][1]);
        
        // Calculate the edge vector and length
        Mat edge = p1 - p0;
        double length = sqrt(edge.dot(edge));
        
        // Divide up the edge
        int numWhiskers = MAX(1, ceil(length/WHISKER_SPACING));
        double spacing = length/(numWhiskers+1);
        
        Mat centres;
        hconcat(p0, p1, centres);
        
        for (int w = 0; w < numWhiskers; w++) {
            Mat centrePt = p0 + edge * ((w+1) * spacing/length);
            hconcat(centres, centrePt, centres);
        }
        
        // Find the projections of the whisker centres
        Mat proj = lsq::projection(pose, centres, K);
        
        // Calculate the length and normal of the edge
        edge = proj.col(1) - proj.col(0);
        length = sqrt(edge.dot(edge));
        Point2f normal = Point2f(edge.at<float>(1), -edge.at<float>(0));
        normal /= length;
        
        for (int w = 0; w < numWhiskers; w++) {
            Point centre = Point(proj.col(2+w).at<float>(0), proj.col(2+w).at<float>(1));
            Mat modelCentre = centres.col(2+w);
            whiskers.push_back(Whisker(centre, normal, modelCentre));
        }
    }
    
    return whiskers;
}

