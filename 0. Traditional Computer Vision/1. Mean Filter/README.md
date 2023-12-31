## Description
The code implements three versions of the following 7x7 mean filter kernel: <br />
```math
\frac{1}{7}\begin{bmatrix}1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\end{bmatrix}
```
While the first version carries out basic 2D convolution of an I/P image, the second and the third versions carry out separable filtering (using 1x7 and 7x1 sub-filters) and separable filtering with a sliding window respectively. The sub-filters are as follows:  <br />
```math
\frac{1}{7}\begin{bmatrix}1 & 1 & 1 & 1 & 1 & 1 & 1\end{bmatrix}
```
```math
\frac{1}{7}\begin{bmatrix}1 & 1 & 1 & 1 & 1 & 1 & 1\end{bmatrix}^T
```
<br />
Although all three versions will yield the exact same O/P image, the execution times and computational complexities will be different. Separating the 7x7 box filter into subfilter kernels in the second version significantly reduces the algorithm's computational complexity. As a sliding window computes the convolution result of a given I/P image pixel using previous sums, its use reduces the number of times the code accesses the memory. As a result, the filter kernel's third version is able to leverage both advantages to ensure both quick execution and reduced computational complexity. <br />

## Results
**512x512 I/P image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/1.%20Mean%20Filter/Results/CV1.jpg) <br /><br />
**Filtered O/P image** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/1.%20Mean%20Filter/Results/CV2.jpg) <br /><br />
**Time complexity of each filter kernel version** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/1.%20Mean%20Filter/Results/Result_1.png) <br /><br />
**Identity of all three O/P image versions**  <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/1.%20Mean%20Filter/Results/Result_2.png) <br /><br />
