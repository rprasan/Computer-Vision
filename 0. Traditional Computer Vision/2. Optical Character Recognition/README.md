## Description
The code implements a matched filter to recognize letters in an I/P image using a given template image. The implementation begins by computing the normalized 8-bit matched-spatial filter (*MSF*) image of the I/P image:

$$MSF[r,c]=\sum_{dr=-W_r/2}^{W_r/2}\sum_{dc=-W_c/2}^{W_c/2}\Bigl[I[r+dr,c+dc]*T[dr+W_r/2,dc+W_c/2]\Bigr]$$

$$MSF_{norm}[r,c]=\frac{255\times\Bigl(MSF[r,c]-min\Bigr)}{max-min}$$

Next, the code loops through a range of detection thresholds *T*. During each iteration of the loop, a small pixel area window slides across the normalized image to employ the current threshold for inspecting all ground truth locations of the letter of interest. If the threshold is not large enough, the inspection process will not detect all occurrences of the letter. In contrast, if the threshold is too large, the inspection process will result in several false detections. Therefore, it is necessary to identify the optimum value of *T* by analyzing the detection results using an ROC curve. <br />

## Results
**I/P image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV1.png) <br /><br />
**Template image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV2.png) <br /><br />
**Normalized MSF image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV3.png) <br /><br />
**ROC curve**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV4.png) <br /><br />
The ROC curve indicates that the optimum value of *T* is 211. <br />
**Threshold images with and without windowing for *T*=211**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV5.png) <br /><br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV6.png) <br /><br />
**Threshold images with and without windowing for *T*=150**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV7.png) <br /><br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/CV8.png) <br /><br />
**Execution terminal**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/2.%20Optical%20Character%20Recognition/Results/ExecutionWindow.png) <br /><br />
