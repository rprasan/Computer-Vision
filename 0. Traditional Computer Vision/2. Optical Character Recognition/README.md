## Description
The code implements a matched filter to recognize letters in an I/P image using a given template image. The implementation begins by computing the normalized 8-bit matched-spatial filter (MSF) image. Next, the code loops through a range of detection thresholds *T*. During each iteration of the loop, a small pixel area centered at the letter ground truth locations on the normalized I/P image is inspected for recognition using the current threshold. The loop iteration finishes when all groud truth locations are covered for all detection threshold values. Finally, an ROC curve is generated to analyze the results and identify the optimum value of *T*.<br />

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
