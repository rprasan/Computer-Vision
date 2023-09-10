## Description
The code tracks an iPhone's motion using accelerometers and gyroscopes. While accelerometers log accelerations along *X*, *Y*, and *Z* directions, gyroscopes log angular velocities along the *pitch*, *roll*, and *yaw* directions. Both sensor data have a sampling frequency of 20Hz and their respective units are gravities (*G*) and *rad/s*. <br /><br />
Tracking the iPhone's motion will require segmenting both sensor data into periods of motion and periods of rest using a window of an appropriate size. The window achieves this by calculating the data variance along each of the six axes. While a large variance indicates a period of motion, a period of rest will correspond to a minuscule variance. <br /><br />

# Raw data: <br />
**Accelerometer data:** <br />
*X*-axis: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/1_1.png) <br />
*Y*-axis: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/1_2.png) <br />
*Z*-axis: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/1_3.png) <br />
All three axes combined: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/1_4.png) <br /><br />
**Gyroscope data:** <br />
Change in *pitch*: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/2_1.png) <br />
Change in *roll*: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/2_2.png) <br />
Change in *yaw*: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/2_3.png) <br />
All three changes combined: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/4.%20Motion%20Tracking/Results/2_4.png) <br /><br />
