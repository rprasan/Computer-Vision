## Description
The code tracks an iPhone's motion using accelerometers and gyroscopes. While accelerometers log accelerations along *X*, *Y*, and *Z* directions, gyroscopes log angular velocities along the *pitch*, *roll*, and *yaw* directions. Both sensor data have a sampling frequency of 20Hz and their respective units are gravities (*G*) and *rad/s*. <br />
Tracking the iPhone's motion will require segmenting both sensor data into periods of motion and periods of rest using a window of an appropriate size. The window achieves this by calculating the data variance along each of the six axes. While a large variance indicates a period of motion, a period of rest will correspond to a minuscule variance. <br />

## Results
**Raw accelerometer data:** <br />
*X*-axis: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam0_World.png) <br />
*Y*-axis: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam1_World.png) <br />
*Z*-axis: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam2_World.png) <br />
All three axes combined: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/Cam3_World.png) <br /><br />
**Raw gyroscope data:** <br />
Change in *pitch*: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam0.PNG) <br />
Change in *roll*: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam1.PNG) <br />
Change in *yaw*: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam2.PNG) <br />
All three changes combined: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/4.%20Camera%20Calibration/Results/1_GridCam3.PNG) <br /><br />
