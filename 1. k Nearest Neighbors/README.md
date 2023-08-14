## _k_ Nearest Neighbors (_kNN_)
Given a test image, the _kNN_ algorithm inspects the labels of the image's _k_ closest training set images and then assigns the majority label among them to the test image. In order to identify the _k_ closest training set images, the algorithm computes pairwise Euclidean distances after unraveling all train and test images into column vectors. Due to this reason, there is a loss of spatial information,  which, in turn, leads to poor performance.  <br /><br />

## Dataset
The dataset used in this experiment is CIFAR-10, which is a collection of 60,000 3x32x32 images, each belonging to one of ten pre-defined labels: _plane_, _car_, _bird_, _cat_, _deer_, _dog_, _frog_, _horse_, _ship_, and _truck_. <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/1.%20k%20Nearest%20Neighbors/Capture.PNG) <br />
