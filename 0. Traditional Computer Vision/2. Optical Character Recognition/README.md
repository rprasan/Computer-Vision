## Description
The code implements a matched filter to recognize letters in an I/P image using a given template image. The implementation begins by computing the normalized 8-bit matched-spatial filter (*MSF*) image of the I/P image:

$$MSF[r,c]=\sum{dr=-W_r/2}{W_r/2}$$

Next, the code loops through a range of detection thresholds *T*. During each iteration of the loop, a small pixel area window slides across the normalized image to employ the current threshold for inspecting all ground truth locations of the letter of interest. If the threshold is not large enough, the inspection process will not detect all occurrences of the letter. In contrast, if the threshold is too large, the inspection process will result in several false detections. Therefore, it is necessary to identify the optimum value of *T* by analyzing the detection results using an ROC curve. <br />

## Results
**I/P image** <br />
![]() <br /><br />
**Template image** <br />
![]() <br /><br />
**Normalized MSF image** <br />
![]() <br /><br />
**ROC curve**  <br />
![]() <br /><br />
**Threshold image for *T*=211**  <br />
![]() <br /><br />
**Threshold image for *T*=150**  <br />
![]() <br /><br />
**Execution terminal**  <br />
![]() <br /><br />
