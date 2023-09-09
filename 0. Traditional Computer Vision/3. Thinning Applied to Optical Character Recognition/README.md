## Description
The code implements thinning, branchpoint, and endpoint detection to the normalized 8-bit matched-spatial filter (*MSF*) image of a given text image to recognize letters in the text. The code loops through a range of detection thresholds *T* and plots an ROC curve to identify the threshold's optimum value. <br />

## Results
**I/P image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/CV1.png) <br /><br />
**Template image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/CV2.png) <br /><br />
**Normalized MSF image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/CV3.png) <br /><br />
**ROC curve: comparison between OCR with and without thinning**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/Image1.jpg) <br /><br />
The ROC curves indicate that the optimum values of *T* with and without thinning are 206 and 211 respectively. <br /><br />
**Thinning example**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/ThinningExample.png) <br /><br />
**Execution terminal**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/3.%20Thinning%20Applied%20to%20Optical%20Character%20Recognition/Results/ExecutionWindow.png) <br /><br />
