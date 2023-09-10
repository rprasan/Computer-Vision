## Description
The goal is to calibrate a camera network using several calibration targets placed in a grid. The calibration process will make use of the Tsai camera calibration model, composed of the following steps, to map each point in the world coordinate system to the pixel coordinate system: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV1.PNG) <br />
**World coordinates to camera coordinates:** <br />
```math
P=RP_W+T
```
```math
\implies\begin{bmatrix}X\\Y\\Z\end{bmatrix}=\begin{bmatrix}r_{11} & r_{12} & r_{13}\\r_{21} & r_{22} & r_{23}\\r_{31} & r_{32} & r_{33}\end{bmatrix}+\begin{bmatrix}X_W\\Y_W\\Z_W\end{bmatrix}+\begin{bmatrix}T_x\\T_y\\T_z\end{bmatrix}
```
where *P*, *R*, *P_W*, and *T* are the coordinates of the point in the camera coordinate system, the rotation matrix, the coordinates of the point in the world coordinate system, and the translation matrix.
**Camera coordinates to ideal undistorted pixel coordinates:** <br />

## Results
**I/P image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/CV1.png) <br /><br />
**Template image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/CV2.png) <br /><br />
**Normalized MSF image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/CV3.png) <br /><br />
**ROC curve: comparison between OCR with and without thinning**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/Image1.jpg) <br /><br />
The ROC curves indicate that the optimum values of *T* with and without thinning are 206 and 211 respectively. In other words, thinning eliminates a lot of 'false positives' to cause the curve to shift to the left, resulting in superior performance.<br /><br />
**Thinning example for the letter *e***  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/ThinningExample.png) <br /><br />
**Execution terminal**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/ExecutionWindow.png) <br /><br />

