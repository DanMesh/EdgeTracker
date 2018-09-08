//
//  area.hpp
//  EdgeTracker
//
//  Created by Daniel Mesham on 05/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef area_hpp
#define area_hpp

#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>

#include "lsq.hpp"
#include "models.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      A library of area matching methods
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
class area {
    
/*
 METHODS
 */
public:
    static double areaError(Vec6f pose, Model * model, Mat img, Mat K);
    static Mat jacobian(Vec6f pose, Model * model, Mat img, Mat K);
    static estimate poseEstimateArea(Vec6f pose1, Model * model, Mat img, Mat K, int maxIter = MAX_ITERATIONS);

/*
 CONSTANTS
 */
public:
    static const int MAX_ITERATIONS = 20;
    static const int ERROR_THRESHOLD = 2;
    
};

#endif /* area_hpp */
