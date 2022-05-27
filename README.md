# Computer Vision
CSE5418 Computer Vision Project / Sogang Univ.
## Project 1
Image Rotation in three methods w/o using `cv2.rotate` [cpp]
* Forward Method
  * For each $(x, y)$, calculate $(x', y')$ by $(x', y') = R(x, y)$
* Backward Method
  * For each $(x', y')$, calculate $(x, y)$ by $(x, y) = R^{-1}(x', y')$
* Backward and Interpolation
  * Backward Method + Interpolation

## Project 2
Check the attached program, analyze detection and matching performance.
* Canny Edge Detection
* Harris Corner Detection

## Project 3
Check the attached SIFT program, analyze computation times of the functions in `computeKeypointsAndDescriptors()`.
