# Description
The code employs 'region growing' to segment an image captured by a range camera into different regions. The range image is grayscale, and the intensity of each pixel in it represents the distance of the corresponding point in the world from the range camera. Thus, the farther a point is from the camera in the world, the brighter the corresponding pixel is in the range image. <br />

The code initially divides the range image into two regions - the background and the foreground - by using an appropriate grayscale pixel intensity threshold. Next, it employs the following expression to compute the surface normal at each world point:
$$\overrightarrow{XA}\times \overrightarrow{XB}=(A-X)\times (B-X)$$
where $X$ is the 3D coordinates of the world point and $A$ and $B$ are the 3D coordinates of the points that are at a fixed pixel distance from the pixel of $X$ along the column and the row directions respectively. <br /><br />
After computing the surface normals, the code identifies seed pixels in the range image and grows regions from them using a $5\times 5$ window. When each region is fully grown, it becomes an image segment. As range image pixel intensities represent distances, a new pixel is added to a growing region based on the deviation angle between the surface normal at that pixel and the average surface normal of the growing region. The pixel is added to the region only if the deviation, as computed by the following dot-product, is less than a fixed threshold:
$$\overrightarrow{A}\cdot \overrightarrow{B}=|a||b|cos(\theta)$$
where $A$ is the 3D coordinates of the average surface normal of the region, $B$ is the 3D coordinates of the surface normal of the new pixel, |a| is the magnitude of $A$, |b| is the magnitude of $B$ and $\theta$ is the deviation. As we are interested in the deviation and not the dot product, the above expression is modified as follows:
$$\theta=cos^{-1}\biggl(\\frac{\overrightarrow{A}\cdot \overrightarrow{B}}{|a||b|}\biggr)$$
When the code's execution reaches a point where no new pixels can be added to a growing region, the region is considered to have fully grown and evolved into an image segment.

# Results:
**Reflectance image of the object:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/5.%20Range%20Image%20Segmentation/Results/CV2.PNG) <br /><br />
**Range image of the object:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/5.%20Range%20Image%20Segmentation/Results/CV1.PNG) <br /><br />
**Threshold image with the background separated from the foreground:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/5.%20Range%20Image%20Segmentation/Results/CV3.PNG) <br /><br />
**Segmented image:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/0.%20Traditional%20Computer%20Vision/5.%20Range%20Image%20Segmentation/Results/CV5.PNG) <br /><br />
