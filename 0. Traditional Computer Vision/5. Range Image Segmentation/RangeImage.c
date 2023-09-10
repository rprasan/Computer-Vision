#include<time.h>
#include<math.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<curses.h>
#define MAX_QUEUE 10000

//declared as global variables as they are used in more than one function. Otherwise, it would be painful to pass them from one function to another
double P[128*128][3];					      //3D coordinates will be in this variable
unsigned char imageCopy[128*128];    	    		      //to store and access the pixels of the source image and its copy
int threshold=128;					      //gray level used to threshold the range image
int distance=2;						      //distance used to compute the cross-product
int intensity=0;					      //to store gray levels used to paint different segments
int count=0;						      //to store the number of pixels in the current region
double surfNorm[128*128][3],avgSurfNorm[3],thetaThreshold=0.7;//surfNorm - stores surface normals of pixels, thetaThreshold - angle threshold used to join new pixels to the region during region growing
							      //avgSurfNorm - stores the average surface normal of the current region during region growing

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////MAIN FUNCTION///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
  system("clear");
  
  FILE *fpt;					             //file pointer for reading the source file
  char *source="/home/rahul/Desktop/ECE 6310/Lab 8/chair-range.ppm";
  unsigned char *image;				             //pointer to store and access the pixels of the source image
  char header[320];		                             //for reading the header of thes ource image
  int i,j,ROWS,COLS,BYTES,r,c,r1,c1;
  int *labels;					             //to store the labels of the segmented regions
  void convert3D(unsigned char*,int,int);		     //function to compute the 3D coordinates of the points corresponding to the pixels
  void RegionGrow(int,int,unsigned char*,int*,int,int);	     //function to grow regions for segmentation
  double angle(double locavgSurfNorm[3],double locsurfNorm[128*128][3],int);//function to find the orientation of the current pixel's surface normal wrt the orientation of the avg surface normal of region
  struct timespec tp1,tp2;			             //to time various operations

  //open the source file for reading
  fpt=fopen(source,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the file for reading!\n");
    exit(0);
  }
  
  //check if the image is in the right format
  i=fscanf(fpt,"%s %d %d %d",header,&ROWS,&COLS,&BYTES);
  if(i!=4 || strcmp(header,"P5")!=0 || BYTES!=255)
  {
    printf("The image is not in the right format!\n");
    exit(0);
  }
  
  //save the pixels of the source image and its copy in the memory
  image=(unsigned char*)calloc(ROWS*COLS,sizeof(unsigned char));
  header[0]=fgetc(fpt);
  fread(image,1,ROWS*COLS,fpt);
  fclose(fpt);					              //source image is no longer needed, close the pointer
  printf("The size of the image is %dX%d.\n\n",ROWS,COLS);
  printf("-----------------------------------------------------------------------------------------\n\n");

////////////////////////////////////////////////////THRESHOLDING///////////////////////////////////////////////////

  //start ticking before threshlding begins
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of thresholding is %lds and %ldns.\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);
  printf("Thresholding begins now...\n");

  //thresholding the range image
  for(i=0;i<ROWS*COLS;i++)
  {
    if(image[i]>threshold)			     	     //threshold used is 128
      image[i]=255;
  }

  //calculate the time taken for thresholding
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Thresholding is complete.\n");
  printf("The time taken for thresholding the image is %lds and %ldns.\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)(abs)(tp2.tv_nsec-tp1.tv_nsec));

  //open the thresholded image file in the write format
  fpt=fopen("RangeImageThresholded.ppm","wb");
  if(fpt==NULL)
  {
    printf("Unable to open the file for writing!\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(image,COLS*ROWS,1,fpt);  
  fclose(fpt);					   	     //writing is now complete, close the file
  printf("Open the file RangeImageThresholded.ppm to see the thresholded image!\n\n");
  printf("-----------------------------------------------------------------------------------------\n\n");

  //creating a copy of the thresholded range image
  for(i=0;i<ROWS*COLS;i++)
    imageCopy[i]=image[i];

///////////////////////////////////////////////////3D COORDINATES//////////////////////////////////////////////////

  //start ticking before calculation begins
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of calculating the 3D coordinates is %lds and %ldns.\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);
  printf("Calculation begins now...\n");

  //call the function convert3D to convert pixels to 3D coordinates
  convert3D(image,ROWS,COLS);

  //calculate the time taken for conversion
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Calculation is complete.\n");
  printf("The time taken for calculating the 3D coordinates is %lds and %ldns.\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)(abs)(tp2.tv_nsec-tp1.tv_nsec));

  //open a file in the write format to write all 3D coordinates
  fpt=fopen("3Dcoords.txt","wb");
  if(fpt==NULL)
  {
    printf("Unable to open the file for writing!\n");
    exit(0);
  }
  for(i=0;i<ROWS*COLS;i++)
  {
    if(i==0)
      fprintf(fpt,"    X\t\t     Y\t\t     Z\n");
    fprintf(fpt,"%f\t%f\t%f\n",P[0][i],P[1][i],P[2][i]);
  }
  printf("Check the file 3Dcords.txt for the results!\n\n");
  printf("-----------------------------------------------------------------------------------------\n\n");

///////////////////////////////////////////////////SURFACE NORMALS/////////////////////////////////////////////////

  double vecXB[3],vecXA[3];				     //to store the coordinates of the vectors XB and XA of each pixel
  for(r=0;r<ROWS*COLS;r++)				     //initializing all elements of the surface normal to zero
  {
    for(i=0;i<3;i++)
      surfNorm[r][i]=0;
  }

  //computing the surface normal - distance between pixels is chosen as 2
  for(r=0;r<ROWS-distance;r++)				     //ignore last 2 (distance=2) rows & columns as we can't extend beyond ROWS-2 along the row direction & COLS-2 along the column direction
  {
    for(c=0;c<COLS-distance;c++)
    {
      for(i=0;i<3;i++)
      {
        vecXB[i]=P[((r+distance)*COLS)+c][i]-P[(r*COLS)+c][i];
	vecXA[i]=P[(r*COLS)+(c+distance)][i]-P[(r*COLS)+c][i];
      }
      surfNorm[(r*COLS+c)][0]=(vecXB[1]*vecXA[2])-(vecXB[2]*vecXA[1]);
      surfNorm[(r*COLS+c)][1]=(vecXB[2]*vecXA[0])-(vecXB[0]*vecXA[2]);
      surfNorm[(r*COLS+c)][2]=(vecXB[0]*vecXA[1])-(vecXB[1]*vecXA[0]);
    }
  }  

////////////////////////////////////////////////////SEGMENTATION///////////////////////////////////////////////////

  //dynamically allocate space to store the labels of the pixels
  labels=(int*)calloc(ROWS*COLS,sizeof(int));

  //assign the label '1' to all background pixels.
  for(i=0;i<ROWS*COLS;i++)				     //The label '1' will be assigned to all pixels that are currently a part of a region. As the background pixels have already been segmented by 
  {							     //assigning 255 to them, they are being assigned the label '1' here itself. The rest of the pixels are not background pixels. So they can be 
    if(image[i]==255)					     //assigned the label '1' only after identifying which region they belong to.
      labels[i]=1;
  }

  //start ticking just before segmentation begins
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Segmentation begins now...\n");
  printf("Time at the beginning of segmentation is %lds and %ldns.\n",(long int)tp1.tv_sec,(long int)tp2.tv_nsec);

  int flag,regions=0;					     //flag - to check if a pixel is a background pixel or not, regions - to store the #regions

  //check if a pixel can be a seed pixel - choose a window of an appropriate size (it's 5X5 here), and ensure there isn't even one background pixel (including the center pixel) in that window
  for(r=2;r<ROWS-2-distance;r++)			     //as the boundary pixels are ignored, the range of 'r' and 'c' are [2,ROWS-2] and [2,COLS-2] respectively. As we must also ignore the last two
  {							     //rows and columns as distance=2, the effective ranges now becomes [2,ROWS-2-distance] and [2,COLS-2-distance] respectively
    for(c=2;c<COLS-2-distance;c++)
    {
      flag=0;
      for(r1=-2;r1<=2;r1++)
      {
        for(c1=-2;c1<=2;c1++)
	{
          if(labels[(r+r1)*128+(c+c1)]==1)		     //background pixel detected, set flag as the current center pixel of the window cannot become a seed pixel
            flag=1;
	}
      }
      if(flag==0)					     //flag is not set. So the current center pixel of the window can be a seed pixel
      {
	regions++;					     //a new region is identified. So increment this variable
        RegionGrow(r,c,image,labels,ROWS,COLS);
        printf("\nThe total #pixels in region %d are %d",regions,count);
        printf("\nThe average surface normal of the region is (%4.2f,%4.2f,%4.2f)\n",avgSurfNorm[0],avgSurfNorm[1],avgSurfNorm[2]);
      }
    }
  }
    printf("\nThere are %d regions in the foreground and + 1 region (wall) in the background.",regions);

  //calculate the time taken for segmentation
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("\n\nSegmentation is complete.\n");
  printf("The time taken for segmenting the image is %lds and %ldns.\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)(abs)(tp2.tv_nsec-tp1.tv_nsec));

  //save the segmented range image as a new image
  fpt=fopen("RangeImageSegmented.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the file RangeImageSegmented.ppm for writing!\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(imageCopy,ROWS*COLS,1,fpt);
  fclose(fpt);

//////////////////////////////////////////////////////COLORING/////////////////////////////////////////////////////

  unsigned char* colorImage;				     //pointer to store and access the pixels of the color image

  //allocate memory dynamically for the color image
  colorImage=(unsigned char*)calloc(ROWS*COLS*3,sizeof(unsigned char));//multiply by 3 as there are three channels. The RGB channels of the first pixel are stored at locations 0, 1 and 2 respectively.
								       //Similarly that of the second pixel at locations 3, 4 and 5 respectively and so on.
  //assigning different colors to different regions
  for(i=0;i<ROWS*COLS;i++)	
  {
    switch(imageCopy[i])
    {
      case 0:colorImage[3*i]=255;			     //RED
    	     colorImage[3*i+1]=0;
    	     colorImage[3*i+2]=0;
	     break;
      case 50:colorImage[3*i]=0;			     //GREEN
    	      colorImage[3*i+1]=255;
    	      colorImage[3*i+2]=0;
	      break;
      case 100:colorImage[3*i]=0;			     //BLUE
	       colorImage[3*i+1]=0;
    	       colorImage[3*i+2]=255;
	       break;
      case 150:colorImage[3*i]=255;			     //YELLOW
	       colorImage[3*i+1]=255;
    	       colorImage[3*i+2]=0;
	       break;
      case 200:colorImage[3*i]=255;			     //MAGENTA
	       colorImage[3*i+1]=0;
    	       colorImage[3*i+2]=255;
	       break;
      case 255:colorImage[3*i]=0;			     //AQUA
   	       colorImage[3*i+1]=255;
    	       colorImage[3*i+2]=255;
	       break;
    }
  }

  //save the segmented range image as a color image
  fpt=fopen("RangeImageSegmentedColor.ppm","wb");
  if(fpt==NULL)
  {
    printf("Unable to open the file RangeImageSegmentedColor.ppm for writing!\n");
    exit(0);
  }
  fprintf(fpt,"P6 %d %d %d\n",COLS,ROWS,255);		     //format of a ppm RGB image
  fwrite(colorImage,3*sizeof(char),ROWS*COLS,fpt);
  fclose(fpt);
  printf("Check the files RangeImageSegmented.ppm and RangeImageSegmentedColor.ppm for the results!\n\n");

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //print the parameters
  printf("The parameters used for segmentation are as follows:\nIntensity threshold = %d\nDistance            = %d\nAngle threshold     = %f\n\n",threshold,distance,thetaThreshold);
  printf("-----------------------------------------------------------------------------------------\n\n");

  return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////MAIN FUNCTION ENDS////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////DOT PRODUCT///////////////////////////////////////////////////

double angle(double locavgSurfNorm[3],double locsurfNorm[128*128][3],int index)//variable name 'avgSurfNorm' not used since avgSurfNorm is a global variable. A local (loc) name is used to avoid conflicts, 
{
  int j=0;
  double dotProduct=0,magAvgSurfNorm=0,magSurfNorm=0;	     //variables to store the dot product and the magnitudes of the average and the current surface normals
  double theta=0;					     //stores the angle of the surface normal of the current pixel wrt to the average surface normal of the current region 

  for(j=0;j<3;j++)
  {
    dotProduct+=locavgSurfNorm[j]*locsurfNorm[index][j];
    magAvgSurfNorm+=locavgSurfNorm[j]*locavgSurfNorm[j];
    magSurfNorm+=locsurfNorm[index][j]*locsurfNorm[index][j];
  }
  magAvgSurfNorm=sqrt(magAvgSurfNorm);			     //magnitude of a vector is the square root of the sum of squares of its constituent elements
  magSurfNorm=sqrt(magSurfNorm);			     //magnitude of a vector is the square root of the sum of squares of its constituent elements
  theta=acos(dotProduct/(magSurfNorm*magAvgSurfNorm));	     //the formula for dot product is A.B=abCos(theta), where 'theta' is the angle of separation between 'A' and 'B'
  return(theta);
}

///////////////////////////////////////////////////REGION GROWING//////////////////////////////////////////////////

void RegionGrow(int r,int c,unsigned char* image,int* labels,int ROWS,int COLS)
{
  int queue[MAX_QUEUE],qh,qt;				     //parameters of the queue
  int i,j,r1,c1;
  int newRow,newCol;					     //used to update the row and column values of the next pixel to be processed inside the queue
  double theta,avgTheta;				     //theta - stores orientation of surface normal of current pixel, avgTheta - stores the average orientation of surface normal of current region
  count=1;						     //to store the number of pixels in the region. Since the center pixel of the window (seed pixel) is already in the region, count starts at 1

  labels[r*128+c]=1;					     //the center pixel of the window when the function RegionGrow is called has already been identified as the seed pixel of the new region
							     //(otherwise this function would not have been called). So it can be safely labelled as '1'.
  //initializing the parameters of the queue
  queue[0]=(r*COLS)+c;					     //first element in the queueu is the seed pixel - center pixel of the window at the time RegionGrow function was called from main()
  qt=0;							     //tail of the queue
  qh=1;							     //head of the queue

  //initialize the average surface normal of the region to that of the seed pixel
  for(i=0;i<3;i++)
    avgSurfNorm[i]=surfNorm[(r*COLS)+c][i];

  //queueing for region growing 
  while(qt!=qh)						     //queue runs until this condition is met. This condition is violated when queue exceeds maximum queue size
  {
    for(r1=-1;r1<=1;r1++)				     //use a 3X3 window as Dr. Hoover's base code for region growing using queueing has a 3X3 window here
    {
      for(c1=-1;c1<=1;c1++)
      {
        if(r1==0 && c1==0)				     //this is the center pixel of the 3X3 window. So do continue
          continue;
        newRow=queue[qt]/128+r1;			     //calculate pixel coordinates of current pixel. As the region grows from the center of the 3X3 window, ensure the growth doesn't include any
        newCol=queue[qt]%128+c1;			     //pixels outside the range of ROWS and COLS of the range image. The range for 'r' & 'c' are [0,ROWS-distance] and [0,COLS-distance] 
        if (newRow<=0 || newRow>=(ROWS-distance) || newCol<=0 || newCol>=(COLS-distance))
          continue;					     //pixel outside the range specified above. So ignore it by doing continue
        if (labels[(newRow)*128+newCol]!=0)		     //the new pixel encountered during growth has already been labelled as belonging to one of the regions. So ignore it by doing continue
          continue;

        //calculate the angle of the surface normal and check if the pixel can be added to the region
	theta=angle(avgSurfNorm,surfNorm,(newRow*128+newCol));
        if(abs(theta)>thetaThreshold)			     //if orientation of surface normal of current pixel outside the limit, ignore it by doing continue.
          continue;
        count++;					     //orientation within the limit, increment count by 1 and the pixel to the region 

	//update the average and the label as the current pixel has been added to the region
	for(i=0;i<3;i++)
          avgSurfNorm[i]=(avgSurfNorm[i]*(count-1)+surfNorm[newRow*128+newCol][i])/count;
        labels[newRow*128+newCol]=1;

	//assign intensity value of current region to pixel corresponding to current pixel in imageCopy
        imageCopy[newRow*128+newCol]=intensity;

	//update queue head to the pixel coordinates of the new region as it is now a part of the region
        queue[qh]=newRow*128+newCol;
        qh=(qh+1)%MAX_QUEUE;
      }
    }
    qt=(qt+1)%MAX_QUEUE;
  }
  intensity+=50;					     //as while loop terminated, the current region has ended. The next region's gray level is current level+50 (ease of visualization)
}

///////////////////////////////////////////////////3D COORDINATES//////////////////////////////////////////////////

void convert3D(unsigned char* image,int ROWS,int COLS)
{
  int  r,c;
  double cp[7],xangle,yangle,dist,ScanDirectionFlag,SlantCorrection;

  //this entire section is copied from the code that Dr. Hoover has provided - it is for finding the 3D coordinates corresponding to each pixel
  cp[0]=1220.7;		/* horizontal mirror angular velocity in rpm */
  cp[1]=32.0;		/* scan time per single pixel in microseconds */
  cp[2]=(COLS/2)-0.5;		/* middle value of columns */
  cp[3]=1220.7/192.0;	/* vertical mirror angular velocity in rpm */
  cp[4]=6.14;		/* scan time (with retrace) per line in milliseconds */
  cp[5]=(ROWS/2)-0.5;		/* middle value of rows */
  cp[6]=10.0;		/* standoff distance in range units (3.66cm per r.u.) */

  cp[0]=cp[0]*3.1415927/30.0;	/* convert rpm to rad/sec */
  cp[3]=cp[3]*3.1415927/30.0;	/* convert rpm to rad/sec */
  cp[0]=2.0*cp[0];		/* beam ang. vel. is twice mirror ang. vel. */
  cp[3]=2.0*cp[3];		/* beam ang. vel. is twice mirror ang. vel. */
  cp[1]/=1000000.0;		/* units are microseconds : 10^-6 */
  cp[4]/=1000.0;			/* units are milliseconds : 10^-3 */

  ScanDirectionFlag=-1;
  for(r=0;r<ROWS;r++)
  {
    for(c=0;c<COLS;c++)
    {
      SlantCorrection=cp[3]*cp[1]*((double)c-cp[2]);
      xangle=cp[0]*cp[1]*((double)c-cp[2]);
      yangle=(cp[3]*cp[4]*(cp[5]-(double)r))+(SlantCorrection*ScanDirectionFlag);
      dist=(double)image[r*COLS+c]+cp[6];
      P[r*COLS+c][2]=sqrt((dist*dist)/(1.0+(tan(xangle)*tan(xangle))+(tan(yangle)*tan(yangle))));
      P[r*COLS+c][0]=tan(xangle)*P[r*COLS+c][2];
      P[r*COLS+c][1]=tan(yangle)*P[r*COLS+c][2];
    }
  }
}
