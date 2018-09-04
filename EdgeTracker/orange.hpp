//
//  orange.hpp
//  LiveTracker
//
//  Created by Daniel Mesham on 01/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef orange_hpp
#define orange_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

class orange {
public:
    static vector<Vec4i> borderLines(Mat img);
    static Mat segmentByColour(Mat img, Scalar colour);
    
};

#endif /* orange_hpp */
