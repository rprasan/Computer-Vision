## Description
The code implements three versions of a 7x7 mean filter kernel. While the first version carries out basic 2D convolution of an I/P image, the second and the third versions carry out separable filtering (using 1x7 and 7x1 sub-filters) and separable filtering with a sliding window respectively.  <br /><br />
Although all three versions will yield the exact same O/P image, the execution times and computational complexities will be different. Separating the 7x7 box filter into subfilter kernels in the second version significantly reduces the algorithm's computational complexity. As a sliding window computes the convolution result of a given I/P image pixel using previous sums, its use reduces the number of times the code accesses the memory. As a result, the filter kernel's third version is able to leverage both advantages to ensure both quick execution and reduced computational complexity. <br />

## Results
