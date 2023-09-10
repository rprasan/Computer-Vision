# Description
The code employs 'region growing' to segment an image captured by a range camera into different regions. The range image is grayscale, and the intensity of each pixel in it represents the distance of the corresponding point in the world from the range camera. Thus, the farther a point is from the camera in the world, the brighter the corresponding pixel is in the range image. <br />

The code initially divides the range image into two regions - the background and the foreground - by using an appropriate grayscale pixel intensity threshold. Next, it employs the following expression to compute the surface normal at each world point:
$$\overrightarrow{XA}\times XB=(A-X)\times (B-X)$$
