## Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I calibrated the camera using 20 chessboard images from the `camera_cal/` directory. The implementation is located in `camera_calibration.py`.

I used `cv2.findChessboardCornersSB()` to detect internal chessboard corners. Since the images show different visible patterns, `find_chessboard_corners()` searches a configurable range of pattern sizes, prioritizing patterns with more corners. All 20 images were processed successfully: 18 yielded a 6×9 pattern, while two yielded a 5×9 pattern.

For each detected pattern, `create_object_points()` generates corresponding 3D coordinates on a flat plane (`z = 0`), assuming unit spacing between adjacent corners. These coordinates are paired with the detected 2D image points and passed to `cv2.calibrateCameraExtended()` through `calibrate_camera()` to estimate the camera matrix and distortion coefficients.

Most images have a resolution of 1280×720, while two have a resolution of 1281×721. The detected coordinates from the latter were scaled to the reference resolution, assuming the difference resulted from resizing. This assumption has not been independently verified.

The calibration achieved an **RMS reprojection error of 0.8553 pixels**. The estimated camera parameters are saved in `camera_parameters.npz` for reuse in the lane detection pipeline.

To demonstrate distortion correction, I applied `cv2.undistort()` to a calibration image. The comparison below shows the original image on the left and the corrected image on the right.

![Original and distortion-corrected calibration image](docs/undistortion_comparison.jpg)

The corrected image shows straighter chessboard lines, particularly near the edges, indicating that the estimated parameters reduce visible lens distortion.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

TODO: Add your text here!!!

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

