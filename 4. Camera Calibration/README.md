## Description
The goal is to calibrate a camera network using several calibration targets placed in a grid. The calibration process will make use of the Tsai camera calibration model, which is composed of the following steps: <br />
1. World coordinates to camera coordinates:
```math
P=RP_W+T
```
```math
\implies\begin{bmatrix}X\\Y\\Z\end{bmatrix}=
```
2. 

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

