//
//  hashing.cpp
//  GeoHash
//
//  Created by Daniel Mesham on 15/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "hashing.hpp"


HashTable hashing::createTable(vector<int> basisID, vector<Point2f> modelPoints, vector<bool> visible, float binWidth) {
    // Creates a hash table using the given point indices to determine the basis points. All remaining model points are then hashed acording to their basis-relative coordinates
    geo_hash gh(binWidth);
    vector<Point2f> basis = { modelPoints[basisID[0]], modelPoints[basisID[1]] };
    vector<bin_index> basisBins;
    
    // Create a list of the bins that correspond to basis points
    for (int i = 0; i < basis.size(); i++) {
        Point2f bc = basisCoords(basis, basis[i]);
        basisBins.push_back(gh.point_to_bin(point(bc.x, bc.y)));
    }
    
    for (int j = 0; j < modelPoints.size(); j++) {
        // Do not hash basis or invisible points
        if (j == basisID[0] || j == basisID[1] || !visible[j]) continue;
        Point2f bc = basisCoords(basis, modelPoints[j]);
        point pt = point(bc.x, bc.y, j);
        // Don't add if in a basis bin
        bool inBin = false;
        for (int b = 0; b < basisBins.size(); b++) {
            if (gh.point_to_bin(pt).equals(basisBins[b])) {
                inBin = true;
                break;
            }
        }
        if (!inBin) gh.add_point( pt );
    }
    
    return HashTable(gh, basisID);
}


vector<HashTable> hashing::voteForTables(vector<HashTable> tables, vector<Point2f> imgPoints, vector<int> imgBasis) {
    // Generates a vote for each table based on how many model points lie in the same bin as a given image point when in the coordinate system of the given basis.
    // Sorts the table based on the number of votes.
    tables = clearVotes(tables);
    
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int i = 0; i < imgPoints.size(); i++) {
        if (i == imgBasis[0] || i == imgBasis[1]) continue;
        Point2f bc = basisCoords(basis, imgPoints[i]);
        point pt = point(bc.x, bc.y);
        
        // Check for matches in each table
        for (int j = 0; j < tables.size(); j++) {
            //vector<point> points = tables[j].table.points_in_bin(pt);
            float w = tables[j].table.width()/2;
            vector<point> points = tables[j].table.points_in_rectangle(point(pt.x() - w, pt.y() - w), point(pt.x() + w, pt.y() + w));
            tables[j].votes += points.size();
            //if (points.size() > 0) tables[j].votes += 1;
        }
    }
    sort(tables.begin(), tables.end(), greater<HashTable>());
    return tables;
}


vector<VoteTally> hashing::voteWithBasis(geo_hash table, vector<Point2f> imgPoints, vector<int> imgBasis) {
    vector<VoteTally> tallies = {};
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int i = 0; i < imgPoints.size(); i++) {
        if (i == imgBasis[0] || i == imgBasis[1]) continue;
        Point2f bc = basisCoords(basis, imgPoints[i]);
        point pt = point(bc.x, bc.y);
        
        //vector<point> points = table.points_in_bin(pt);
        float w = table.width()/2;
        vector<point> points = table.points_in_rectangle(point(pt.x() - w, pt.y() - w), point(pt.x() + w, pt.y() + w));
        
        for (int p = 0; p < points.size(); p++) {
            model_basis mb = points[p].modelBasis();
            bool voted = false;
            for (int t = 0; t < tallies.size(); t++) {
                if (tallies[t].mb == mb) {
                    tallies[t].vote();
                    voted = true;
                    break;
                }
            }
            if (!voted) {
                VoteTally vt = VoteTally(mb);
                vt.vote();
                tallies.push_back(vt);
            }
        }
    }
    sort(tallies.begin(), tallies.end(), greater<VoteTally>());
    return tallies;
}


vector<HashTable> hashing::clearVotes(vector<HashTable> tables) {
    for (int i = 0; i < tables.size(); i++) {
        tables[i].votes = 0;
    }
    return tables;
}


vector<Mat> hashing::getOrderedPoints(vector<int> imgBasis, HashTable ht, vector<Point3f> modelPoints, vector<Point2f> imgPoints) {
    // Returns a Mat of model points and image points for use in the least squares algorithm. The orders of both are the same (i.e. the i-th model point corresponds to the i-th image point).
    vector<int> basisIndex = ht.basis;
    vector<Point3f> orderedModelPoints;
    vector<Point2f> orderedImgPoints;
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int j = 0; j < imgPoints.size(); j++) {
        Point2f bc = basisCoords(basis, imgPoints[j]);
        
        // If a basis point...
        if (j == imgBasis[0]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[0]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        else if (j == imgBasis[1]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[1]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        
        // If not a basis point...
        else {
            point pt = point(bc.x, bc.y);
            vector<point> binPoints = ht.table.points_in_bin(pt);
            
            if (binPoints.size() > 0) {
                // Take the first point in the bin
                int modelPt_ID = binPoints[0].getID();
                
                orderedModelPoints.push_back(modelPoints[modelPt_ID]);
                orderedImgPoints.push_back(imgPoints[j]);
            }
        }
    }
    
    Mat newModel = pointsToMat3D_Homog(orderedModelPoints);
    Mat imgTarget = pointsToMat2D(orderedImgPoints).t();
    return {newModel, imgTarget};
}


Point2f hashing::basisCoords(vector<Point2f> basis, Point2f p) {
    // Converts the coordinates of point p into the reference frame with the given basis
    Point2f O = (basis[0] + basis[1])/2;
    basis[0] -= O;
    basis[1] -= O;
    p = p - O;
    
    float B = sqrt(pow(basis[1].x, 2) + pow(basis[1].y, 2));
    float co = basis[1].x / B;
    float si = basis[1].y / B;
    
    float u =  co * p.x + si * p.y;
    float v = -si * p.x + co * p.y;
    
    return Point2f(u, v)/B;
}

Mat hashing::pointsToMat2D(vector<Point2f> points) {
    int rows = 2;
    int cols = int(points.size());
    
    float table[rows][cols];
    for (int c = 0; c < cols; c++) {
        table[0][c] = points[c].x;
        table[1][c] = points[c].y;
    }
    return Mat(rows, cols, CV_32FC1, table) * 1;
}

Mat hashing::pointsToMat3D(vector<Point3f> points) {
    int rows = 3;
    int cols = int(points.size());
    
    float table[rows][cols];
    for (int c = 0; c < cols; c++) {
        table[0][c] = points[c].x;
        table[1][c] = points[c].y;
        table[2][c] = points[c].z;
    }
    return Mat(rows, cols, CV_32FC1, table) * 1;
}

Mat hashing::pointsToMat3D_Homog(vector<Point3f> modelPoints) {
    // Converts 3D model points into their full 3D homogeneous representation
    Mat m = pointsToMat3D(modelPoints);
    
    Mat one = Mat::ones(1, m.cols, CV_32FC1);
    vconcat(m, one, m);
    
    return m * 1;
}

geo_hash hashing::hashModelsIntoTable(geo_hash table, vector<Model *> model, Mat K, float defaultZ) {
    // Take the given models and hash their points into the given hash table
    // Uses all the vertex  pairs in the model as possible basis points
    
    float dA = 2 * CV_PI / table.numRotBins;    // Bin width
    
    for (int xBin = 0; xBin < table.numRotBins; xBin++) {
        float angleX = dA * (0.5 + xBin);
        for (int yBin = 0; yBin < table.numRotBins/2; yBin++) {
            float angleY = (dA * (0.5 + yBin)) - CV_PI/2;
            Vec6f pose = {0, 0, defaultZ, angleX, angleY, 0};
            vector<int> view = {xBin, yBin};
            
            for (int m = 0; m < model.size(); m++) {
                vector<vector<int>> basisList = model[m]->getEdgeBasisList();
                
                Mat proj = lsq::projection(pose, model[m]->pointsToMat(), K);
                
                // Convert Mat of points to point objects
                vector<Point2f> modelPoints2D;
                for (int i  = 0; i < proj.cols; i++) {
                    modelPoints2D.push_back(Point2f(proj.at<float>(0,i), proj.at<float>(1,i)));
                }
                
                vector<bool> vis = model[m]->visibilityMask(angleX, angleY);
                
                for (int i = 0; i < basisList.size(); i++) {
                    vector<int> basisIndex = basisList[i];
                    
                    // Don't make a hash table if the basis isn't visible
                    if (!vis[basisIndex[0]] || !vis[basisIndex[1]]) continue;
                    
                    // Fill in createTable
                    vector<Point2f> basis = { modelPoints2D[basisIndex[0]], modelPoints2D[basisIndex[1]] };
                    vector<bin_index> basisBins;
                    
                    // Create a list of the bins that correspond to basis points
                    for (int j = 0; j < basis.size(); j++) {
                        Point2f bc = basisCoords(basis, basis[j]);
                        basisBins.push_back(table.point_to_bin(point(bc.x, bc.y)));
                    }
                    
                    for (int k = 0; k < modelPoints2D.size(); k++) {
                        // Do not hash basis or invisible points
                        if (k == basisIndex[0] || k == basisIndex[1] || !vis[k]) continue;
                        Point2f bc = basisCoords(basis, modelPoints2D[k]);
                        point pt = point(bc.x, bc.y, k, model_basis(m, basisIndex, view));
                        // Don't add if in a basis bin
                        bool inBin = false;
                        for (int b = 0; b < basisBins.size(); b++) {
                            if (table.point_to_bin(pt).equals(basisBins[b])) {
                                inBin = true;
                                break;
                            }
                        }
                        if (!inBin) table.add_point( pt ); cout << "*";
                    }
                    cout << endl;
                }
            }
        }
    }
    
    
    
    for (int m = 0; m < model.size(); m++) {
        
        vector<Point3f> modelPoints3D = model[m]->getVertices();
        
        // Convert 3D coordinates to 2D
        vector<Point2f> modelPoints2D;
        for (int i = 0; i < modelPoints3D.size(); i++) {
            modelPoints2D.push_back( Point2f(modelPoints3D[i].x, modelPoints3D[i].y) );
        }
        
        // A list of model basis pairs
        vector<vector<int>> basisList = model[m]->getEdgeBasisList();
        
        // Get the visibility mask (should be all true)
        vector<bool> vis = model[m]->visibilityMask(0, 0);
        
        for (int i = 0; i < basisList.size(); i++) {
            vector<int> basisIndex = basisList[i];
            
            // Don't make a hash table if the basis isn't visible
            if (!vis[basisIndex[0]] || !vis[basisIndex[1]]) continue;
            
            // Fill in createTable
            vector<Point2f> basis = { modelPoints2D[basisIndex[0]], modelPoints2D[basisIndex[1]] };
            vector<bin_index> basisBins;
            
            // Create a list of the bins that correspond to basis points
            for (int j = 0; j < basis.size(); j++) {
                Point2f bc = basisCoords(basis, basis[j]);
                basisBins.push_back(table.point_to_bin(point(bc.x, bc.y)));
            }
            
            for (int k = 0; k < modelPoints2D.size(); k++) {
                // Do not hash basis or invisible points
                if (k == basisIndex[0] || k == basisIndex[1] || !vis[k]) continue;
                Point2f bc = basisCoords(basis, modelPoints2D[k]);
                point pt = point(bc.x, bc.y, k, model_basis(m, basisIndex));
                // Don't add if in a basis bin
                bool inBin = false;
                for (int b = 0; b < basisBins.size(); b++) {
                    if (table.point_to_bin(pt).equals(basisBins[b])) {
                        inBin = true;
                        break;
                    }
                }
                if (!inBin) table.add_point( pt ); cout << "*";
            }
            cout << endl;
        }
        cout << "^ Model ^" << endl << endl;
    }
    return table;
}

vector<Mat> hashing::getOrderedPoints2(geo_hash table, model_basis mb, vector<int> imgBasis, vector<Point3f> modelPoints, vector<Point2f> imgPoints) {
    // Returns a Mat of model points and image points for use in the least squares algorithm. The orders of both are the same (i.e. the i-th model point corresponds to the i-th image point).
    vector<int> basisIndex = mb.basis;
    vector<Point3f> orderedModelPoints;
    vector<Point2f> orderedImgPoints;
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int j = 0; j < imgPoints.size(); j++) {
        Point2f bc = basisCoords(basis, imgPoints[j]);
        
        // If a basis point...
        if (j == imgBasis[0]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[0]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        else if (j == imgBasis[1]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[1]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        
        // If not a basis point...
        else {
            point pt = point(bc.x, bc.y);
            vector<point> binPoints = table.points_in_bin(pt);
            
            for (int p = 0; p < binPoints.size(); p++) {
                if (binPoints[p].modelBasis() == mb) {
                    int modelPt_ID = binPoints[p].getID();
                    orderedModelPoints.push_back(modelPoints[modelPt_ID]);
                    orderedImgPoints.push_back(imgPoints[j]);
                    break;
                }
            }
        }
    }
    
    Mat newModel = pointsToMat3D_Homog(orderedModelPoints);
    Mat imgTarget = pointsToMat2D(orderedImgPoints).t();
    return {newModel, imgTarget};
}

float hashing::gaussianVote(point centre, point pt, float sigma) {
    // Returns the vote for a point based on the given centre,
    // assuming a Gaussian probablility distribution with
    // standard deviation = sigma
    float dx = centre.x() - pt.x();
    float dy = centre.y() - pt.y();
    float sqErr = dx*dx + dy*dy;
    
    return exp(- sqErr / (2 * sigma * sigma));
}
