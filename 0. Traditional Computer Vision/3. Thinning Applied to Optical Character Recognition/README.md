## Description
The code implements thinning, branchpoint, and endpoint detection to the normalized 8-bit matched-spatial filter (*MSF*) image of a given text image to recognize letters in the text. The code loops through a range of detection thresholds *T* and plots an ROC curve to identify the threshold's optimum value. <br />

## Results
**I/P image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV1.png) <br /><br />
**Template image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV2.png) <br /><br />
**Normalized MSF image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV3.png) <br /><br />
**ROC curve**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV4.png) <br /><br />
The ROC curve indicates that the optimum value of *T* is 211. <br /><br />
**Threshold images with and without windowing for *T*=211**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV5.png) <br /><br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV6.png) <br /><br />
**Threshold images with and without windowing for *T*=150**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV7.png) <br /><br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV8.png) <br /><br />
**Execution terminal**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/ExecutionWindow.png) <br /><br />

