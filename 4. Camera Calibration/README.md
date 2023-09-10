## Description
The goal is to calibrate a camera network consisting of 6 cameras using several calibration targets placed in a grid. The calibration process will make use of the Tsai camera calibration model, composed of the following steps, to map each point in the world coordinate system to the pixel coordinate system: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV7.PNG) <br /><br />
**World coordinates to camera coordinates:** <br />
```math
P=RP_W+T
```
```math
\implies\begin{bmatrix}X\\Y\\Z\end{bmatrix}=\begin{bmatrix}r_{11} & r_{12} & r_{13}\\r_{21} & r_{22} & r_{23}\\r_{31} & r_{32} & r_{33}\end{bmatrix}+\begin{bmatrix}X_W\\Y_W\\Z_W\end{bmatrix}+\begin{bmatrix}T_x\\T_y\\T_z\end{bmatrix}
```
where $P$, $R$, $P_W$, and $T$ are the coordinates of the point in the camera coordinate system, the rotation matrix, the coordinates of the point in the world coordinate system, and the translation matrix. <br /><br />
**Camera coordinates to ideal undistorted pixel coordinates:** <br />
Applying the concept of similarity of triangles to the below image yields:
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV8.PNG) <br />
$$\frac{X}{Z}=\frac{X_u}{f}$$
$$\frac{Y}{Z}=\frac{Y_u}{f}$$
$$\implies X_u=f\frac{X}{Z}$$
$$\implies Y_u=f\frac{Y}{Z}$$
where $X_u$ and $Y_u$ are the undistorted pixel coordinates of the point and $f$ if the camera's focal distance. <br /><br />
**Ideal undistorted pixel coordinates to real (distorted) pixel coordinates:** <br />
Radial distortions, as illustrated in the below figure, can be corrected by employing the following mathematical expressions: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV9.PNG) <br />
$$X_d=\frac{X_u}{1+k_1r^{2}+k_2r^{4}}$$
$$Y_d=\frac{Y_u}{1+k_1r^{2}+k_2r^{4}}$$
where $X_d$ and $Y_d$ are the point's distorted pixel coordinates and $r=\sqrt{X_u^{2}+Y_u^{2}}$. <br /><br />
**Real pixel coordinates to computer digitized coordinates** <br />
The following expressions help digitize and store the point's real pixel coordinates in a computer's physical memory: <br />
$$X_f=S_X\frac{N_{fX}}{N_{cX}}\frac{1}{dX}X_d+C_X$$
$$Y_f=\frac{1}{dY}Y_d+C_Y$$
where $C_X$ and $C_Y$ denote the center of the frame in the hardware, $\frac{N_{fX}}{N_{cX}}$ is the #sensor elements in the CCD along the *x*-direction, $\frac{1}{dX}$ and $\frac{1}{dY}$ are the physical distances between the sensor elements and $S_X$ is the difference in the separation between consecutive elements along the *x* and *y*-directions. <br /><br />
## Procedure
**The world as seen by the cameras:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam0_World.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam1_World.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam2_World.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam3_World.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam4_World.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam5_World.png) <br /><br />
**The grids as detected by the cameras:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam0.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam1.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam2.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam3.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam4.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam5.PNG) <br /><br />
**Undistorted images:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV1.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV2.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV3.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV4.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV5.PNG) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/CV6.PNG) <br /><br />
**Tracking objects using the calibrated camera network:** <br />
Region of interest:
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/3_Tracking1.PNG) <br />
Tracking both a human and a chair:
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/3_Tracking2.PNG) <br />
Tracking a human when the pixel brightness threshold is 150:
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/3_Tracking3.PNG) <br />
Tracking a human when the pixel brightness threshold is 170:
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/3_Tracking4.PNG) <br />
