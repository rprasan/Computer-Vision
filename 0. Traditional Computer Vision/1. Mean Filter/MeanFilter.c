#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<string.h>

int main()
{
  FILE *fpt;				//file pointer
  unsigned char *image;			//image pointer for source image
  unsigned char	*smoothed_image;	//image pointer for target image
  int *intermediate_image;		//image pointer for intermediate image
  char header[320];
  int ROWS,COLS,BYTES,i,r,c,r1,c1,sum,sum1;
  struct timespec tp1,tp2;		//for the timer

  //open the source image in the read format
  fpt=fopen("bridge.ppm","rb");
  if (fpt==NULL)
  {
    printf("Unable to open the file bridge.ppm for reading\n");
    exit(0);
  }

  //check if the image is in the right format (PPM)
  i=fscanf(fpt,"%s %d %d %d",header,&COLS,&ROWS,&BYTES);
  printf("i=%d\n",i);
  if (i!=4 || strcmp(header,"P5") != 0  ||  BYTES != 255)
  {
    printf("The file is not an 8-bit greyscale PPM-format image\n");
    exit(0);
  }

  //save pixels of the source image in the memory
  image=(unsigned char *)calloc((ROWS*COLS),sizeof(unsigned char));
  header[0]=fgetc(fpt);
  fread(image,1,COLS*ROWS,fpt);
  fclose(fpt);				//source image no longer needed, close the file

  //allocate memory for the pixels of the target image
  smoothed_image=(unsigned char *)calloc((ROWS*COLS),sizeof(unsigned char));


/////////////////////////////////////BASIC 2D CONVOLUTION///////////////////////////////


  //start ticking just before smoothing begins
  printf("Basic 2D convolution\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of the operation is %lds and %lds\n",(long int)tp1.tv_sec,tp1.tv_nsec);

  //smoothing begins
  for(r=3;r<ROWS-3;r++)
  {
    for(c=3;c<COLS-3;c++)
    {
      sum=0;
      for(r1=-3;r1<=3;r1++)
      {
        for(c1=-3;c1<=3;c1++)
          sum+=image[((r+r1)*COLS)+(c+c1)];
      }
      smoothed_image[(r*COLS)+c]=sum/49;//7X7 kernel has 49 elements
    }
  }

  //compute time taken for smoothing
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of the operation is %lds and %ldns\n",(long int)tp2.tv_sec,tp2.tv_nsec);
  printf("Time taken to smoothen the image is %lds and %ldns\n", (tp2.tv_sec-tp1.tv_sec), (tp2.tv_nsec-tp1.tv_nsec));

  //open the target image in the write format
  fpt=fopen("smoothenM7_v1.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the file smoothenM7_v1.ppm for writing\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(smoothed_image,COLS*ROWS,1,fpt);
  fclose(fpt);//writing is now complete, close the target image
  printf("Round 1 of smoothing is now complete\n......................................................................\n");



////////////////////////////////////SEPARABLE KERNELS//////////////////////////////////


  //allocate memory for the pixels of the intermediate image
  intermediate_image=(int *)calloc((ROWS*COLS),sizeof(int));//declare as int since this image being intermediate in nature, will be used for computation involving number

  //start ticking just before smoothing begins
  printf("Separable filter kernels\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of the operation is %lds and %lds\n",(long int)tp1.tv_sec,tp1.tv_nsec);

  //smoothing begins
  for(r=0;r<ROWS;r++)//fc kernel begins here
  {
    for(c=3;c<COLS-3;c++)
    {
      sum=0;
      for(c1=-3;c1<=3;c1++)
        sum+=image[(r)*COLS+(c+c1)];
      intermediate_image[(r*COLS)+c]=sum;
    }
  }		     //fc kernel ends
  for(c=0;c<ROWS;c++)//fr kernel begins
  {
    for(r=3;r<COLS-3;r++)
    {
      sum=0;
      for(r1=-3;r1<=3;r1++)
        sum+=intermediate_image[((r+r1)*COLS)+c];
      smoothed_image[(r*COLS)+c]=sum/49;
    }
  }		    //fr kernel ends

  //compute time taken for smoothing
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of the operation is %lds and %ldns\n\n",(long int)tp2.tv_sec,tp2.tv_nsec);
  printf("Time taken to smoothen the image is %lds and %ldns\n", (tp2.tv_sec-tp1.tv_sec), (tp2.tv_nsec-tp1.tv_nsec));

  //open the target image in the write format
  fpt=fopen("smoothenM7_v2.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the file smoothenM7_v2.ppm for writing\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(smoothed_image,COLS*ROWS,1,fpt);
  fclose(fpt);//writing is now complete, close the target image
  printf("Round 2 of smoothing is now complete\n......................................................................\n");



////////////////////////////SEPARABLE KERNELS AND SLIDING WINDOW///////////////////////


  //start ticking just before smoothing begins
  printf("Separable filter kernels with sliding window\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of the operation is %lds and %lds\n",(long int)tp1.tv_sec,tp1.tv_nsec);

  //smoothing begins
  for(r=0;r<ROWS;r++)//fc kernel begins here
  {
    sum=0;
    for(c=3;c<COLS-3;c++)
    {
      sum1=0;
      if(c==3)
      {
        for(c1=-3;c1<=3;c1++)
          sum+=image[(r*COLS)+(c+c1)];
	sum1=sum;
      }
      else
      {
        sum1=sum-image[(r*COLS)+(c-4)]+image[(r*COLS)+(c+3)];
        sum=sum1;
      }
      intermediate_image[r*COLS+c]=sum1;
    }
  }		     //fc kernel ends
  for(c=0;c<ROWS;c++)//fr kernel begins here
  {
    sum=0;
    for(r=3;r<COLS-3;r++)
    {
      sum1=0;
      if(r==3)
      {
        for(r1=-3;r1<=3;r1++)
	  sum+=intermediate_image[((r+r1)*COLS)+c];
	sum1=sum;
      }
      else
      {
        sum1=sum-intermediate_image[((r-4)*COLS)+c]+intermediate_image[((r+3)*COLS)+c];
        sum=sum1;
      }
      smoothed_image[(r*COLS)+c]=sum1/49;
    }
  }		     //fr kernel ends

  //compute time taken for smoothing
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of the operation is %lds and %ldns\n\n",(long int)tp2.tv_sec,tp2.tv_nsec);
  printf("Time taken to smoothen the image is %lds and %ldns\n", (tp2.tv_sec-tp1.tv_sec), (tp2.tv_nsec-tp1.tv_nsec));

  //open the target image in the write format
  fpt=fopen("smoothenM7_v3.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the file smoothenM7_v3.ppm for writing\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(smoothed_image,COLS*ROWS,1,fpt);
  fclose(fpt);//writing is now complete, close the target image  
}
