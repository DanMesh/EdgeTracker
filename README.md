# EdgeTracker
An edge-based implementation for model-based visual tracking, inspired by the methods used in Active Shape Model systems

The second attempt at a complete model-based tracking system for my EEE4022S thesis project.

## OpenCV Setup

### 1. Download OpenCV
   ```
   brew install pkg-config
   brew install opencv
   ```

### 2. Set XCode search paths
   In the project's Build Settings:
   - Set 'Header Search Paths' to `/usr/local/Cellar/opencv/3.4.2/include`
   - Set 'Library Search Paths' to `/usr/local/Cellar/opencv/3.4.2/lib`

### 3. Add the relevant frameworks to the project
   - Right click on the project in the naviagtor and select "Add Files to [Project]..."
   - Navigate to `/usr/local/Cellar/opencv/3.4.2/lib/`
   - Add the following files:
     ```
     libopencv_core.3.4.2.dylib
     libopencv_highgui.3.4.2.dylib
     libopencv_imgcodecs.3.4.2.dylib
     etc...
     ```
     (choose files as per the included libraries in `main.cpp`)
