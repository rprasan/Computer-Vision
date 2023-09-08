#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curses.h>

int main()
{
  system("clear");				    //clear the screen - included in curses.h

  FILE *fpt;					    //file pointer for reading the image file
  char* source="/home/rahul/Desktop/ECE 6310/Lab 2/parenthood.ppm";
  char* templ="/home/rahul/Desktop/ECE 6310/Lab 2/parenthood_e_template.ppm";
  char* gt="/home/rahul/Desktop/ECE 6310/Lab 2/parenthood_gt.txt";
  unsigned char *image,*template,*imageMSF,*imageTh;//pointer to store and access source, template and MSF image pixels
  int *zmtemplate;		  	  	    //pointer to store and acess zero-mean template pixels-should be int, otherwise would not handle negative values in the zero-mean template
  long int *intermediate_image;			    //pointer to store and access intermediate image pixels
  char header[320],header_T[320];
  int ROWS,COLS,BYTES,i,ROWS_T,COLS_T,BYTES_T,r,c,mean=0,r1,c1;
  int gt_c[1262],gt_r[1262],gtr[1262][2];	    //row and column information of ground truth
  char gt_ch[1262];			  	    //character information of ground truth
  int count=0;					    //to save #appearances of 'e'
  long int sum=0,max=0,min=0,range;		    //for nomalizing MSF image to 8 bits
  int threshold;				    //for thresholding the output image of MSF
  int metric[256][4];		    		    //1st column is TP, 2nd is FP, 3rd is FN and 4th is TN
  float TPR[256],FPR[256],accuracy[256];	    //true positive rates, false positive rates and accuracies for different threshold levels
  int checksum=0;				    //to check for the presence of 'e' in the window
  bool check;					    //to check if letter is 'e' or not
  int thresholdIdeal=0;				    //to store the ideal threshold value
  struct timespec tp1,tp2;			    //for timing the operations
//////////////////////////////////////////////////SOURCE IMAGE//////////////////////////////////////////////////
  //open the source image in the read format
  fpt=fopen(source,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the image parenthood.ppm for reading\n");
    exit(0);
  }
  
  //check if the image is in the correct format (PPM)
  i=fscanf(fpt,"%s %d %d %d",header,&COLS,&ROWS,&BYTES);
  if(i!=4 || strcmp(header,"P5")!=0 || BYTES!=255)
  {
    printf("i=%d\n",i);
    printf("The file is not an 8-bit grayscale PPM-format image\n");
    exit(0);
  }

  //save the pixels of the source image in the memory
  image=(unsigned char *)calloc(ROWS*COLS,sizeof(unsigned char));
  header[0]=fgetc(fpt);
  fread(image,1,ROWS*COLS,fpt);
  fclose(fpt);					   //source image file is no longer needed, close it
////////////////////////////////////////////////////TEMPLATE////////////////////////////////////////////////////
  //open the template in the read format
  fpt=fopen(templ,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the template for reading\n");
    exit(0);
  }

  //check if the template is in the correct format(PPM)
  i=fscanf(fpt,"%s %d %d %d",header_T,&COLS_T,&ROWS_T,&BYTES_T);
  if(i!=4 || strcmp(header_T,"P5")!=0 || BYTES_T!=255)
  {
    printf("i=%d\n",i);
    printf("The template is not an 8-bit grayscale PPM-format image\n");
    exit(0);
  }

  //save the template in the memory
  template=(unsigned char *)calloc(ROWS_T*COLS_T,sizeof(unsigned char));
  header_T[0]=fgetc(fpt);
  fread(template,1,ROWS_T*COLS_T,fpt);
  fclose(fpt);					  //template file is no longer needed, close it
//////////////////////////////////////////////////GROUND TRUTH//////////////////////////////////////////////////
  //open the ground truth file for reading
  fpt=fopen(gt,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the ground truth file for reading\n");
    exit(0);
  }
  
  //save the contents of the ground truth file
  for(i=0;;i++)
  {
    fscanf(fpt,"%c %d %d\n",&gt_ch[i],&gtr[i][0],&gtr[i][1]);
    if(gt_ch[i]=='\0')
      break;
    if(gt_ch[i]=='e')
      count++;
  }
  fclose(fpt);					  //ground truth file is no longer needed, close it
  printf("The #e's (ground truth) in the source image is %d\n\n",count);
////////////////////////////////////////////////ZERO-MEAN TEMPLATE//////////////////////////////////////////////  
  //allocate memory for the pixels of the zero-mean template
  zmtemplate=(int *)calloc(ROWS_T*COLS_T,sizeof(int));

  //compute the mean of the template pixel values
  for(r=0;r<ROWS_T;r++)
  {
    for(c=0;c<COLS_T;c++)
      mean+=(int)template[(r*COLS_T)+c];
  }
  mean=(mean)/(ROWS_T*COLS_T);
  printf("Template mean is %d\n\n",mean);

  //compute the pixels of the zero-mean template
  for(r=0;r<ROWS_T;r++)
  {
    for(c=0;c<COLS_T;c++)
    {
      zmtemplate[(r*COLS_T)+c]=(int)(template[(r*COLS_T)+c]-mean);//gcc handles implicit type conversion from unsigned char to int
      //printf("%d ",zmtemplate[(r*COLS_T)+c]);		   	  //print the zero-mean template
    }
    //printf("\n");
  }

  //open the zero-mean template image in the write format
  fpt=fopen("Zero-meanTemplate.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the image Zero-meanTemplate.ppm for writing\n");
    exit(0); 
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS_T,ROWS_T);
  fwrite(zmtemplate,COLS_T*ROWS_T,1,fpt);
  fclose(fpt);					  //writing is now complete, close the file
///////////////////////////////////////////////MATCHED SPATIAL FILTER//////////////////////////////////////////////
  //allocate memory for the pixels of the intermediate image
  intermediate_image=(long int *)calloc(ROWS*COLS,sizeof(long int));

  //start ticking just before MSF begins
  printf("Filtering begins now...\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of filtering is %lds and %ldns\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);

  //perform cross-correlation using the zero-mean template
  for(r=7;r<ROWS-7;r++)	  //same logic as lab #1 - floor((rows size)/2)
  {
    for(c=4;c<COLS-4;c++) //same logic as lab #1 - floor((column size)/2)
    {
      sum=0;
      for(r1=-7;r1<=7;r1++)
      {
        for(c1=-4;c1<=4;c1++)
          sum+=(long int)image[((r+r1)*COLS)+(c+c1)]*(long int)zmtemplate[((7+r1)*COLS_T)+(4+c1)];
      }
      intermediate_image[(r*COLS)+c]=sum;
      if(max<sum)	  //updating maximum pixel value for normalization
        max=sum;
      if(min>sum)	  //updating minimum pixel value for normalization
        min=sum;
    }
  }

  //allocate memory for the pixels of the output image of MSF
  imageMSF=(unsigned char *)calloc(ROWS*COLS,sizeof(unsigned char));
  
  //perform normalization to convert MSF image to grayscale. Use the formula ((pixel_value-min)/range)*255
  range=max-min;
  for(r=7;r<ROWS-7;r++)
  {
    for(c=4;c<COLS-4;c++)
      imageMSF[(r*COLS)+c]=(unsigned char)255*(intermediate_image[(r*COLS)+c]-min)/range;
  }

  //compute time taken for filtering
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of filtering is %lds and %ldns\n",(long int)tp2.tv_sec,tp2.tv_nsec);
  printf("Time taken to filter the image is %lds and %ldns\n\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)(tp2.tv_nsec-tp1.tv_nsec));

  //open the MSF image file in the write format
  fpt=fopen("MSFimage.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the image MSFimage for writing\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(imageMSF,ROWS*COLS,1,fpt);
  fclose(fpt);			//writing is now complete, close the file
////////////////////////////////////////////////////THRESHOLDING///////////////////////////////////////////////////
  //start ticking just before thresholding begins
  printf("Thresholding begins now...\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of thresholding is %lds and %ldns\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);

  //thresholding begins here
  for(threshold=0;threshold<256;threshold++)
  {
    for(i=0;i<4;i++)		//to set TP, FP, FN and TN of the current threshold value to zeros
      metric[threshold][i]=0;
    for(i=0;i<1262;i++)
    {
      if(gt_ch[i]=='e')		//check if ground truth is 'e' or not
        check=true;
      else
        check=false;
      for(r=gtr[i][1]-7;r<=gtr[i][1]+7;r++)
      {
        for(c=gtr[i][0]-4;c<=gtr[i][0]+4;c++)
        {
          if(imageMSF[(r*COLS)+c]>threshold)
            checksum++;	       //increments every time a pixel value in the window is greater than the current threshold 
        }
      }
      if(checksum>0)	       //if checksum>0 then there is at least one pixel with a value greater than the threshold. All such cases are positives - both TP and FP
      {
        if(check==true)	       //checking if a positive is TP or not
          metric[threshold][0]++;
        else
          metric[threshold][1]++;
      }
      else		       //if checksum=0 then there is no pixel with a value greater than the threshold. All such cases are negatives - both FN and TN
      {
        if(check==true)	       //checking if a negative is TN or not
          metric[threshold][2]++;
        else
          metric[threshold][3]++;
      }
      checksum=0;
    }
    TPR[threshold]=(float)metric[threshold][0]/(metric[threshold][0]+metric[threshold][2]);	//TPR=TP/(TP+FN)
    FPR[threshold]=(float)metric[threshold][1]/(metric[threshold][1]+metric[threshold][3]);	//FPR=FP/(FP+TN)
    accuracy[threshold]=(float)metric[threshold][0]/(metric[threshold][0]+metric[threshold][1]);//accuracy=TP/(TP+FP)
  }

  //computing the time taken for thresholding
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of thresholding is %lds and %ldns\n",(long int)tp2.tv_sec,(long int)tp2.tv_nsec);
  printf("Time taken to threshold the image is %lds and %ldns\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)(tp2.tv_sec-tp1.tv_sec));

  //open a file in the write format to write all metric values
  fpt=fopen("Metrics.txt","w");
  for(threshold=0;threshold<256;threshold++)
  {
    if(threshold==0)	       //to write the heading of the file
      fprintf(fpt,"TP\tFP\tFN\tTN\tTPR\tFPR\tAcc\n");
    fprintf(fpt,"%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\n",metric[threshold][0],metric[threshold][1],metric[threshold][2],metric[threshold][3],TPR[threshold],FPR[threshold],accuracy[threshold]);
  }
  fclose(fpt);		      //writing is now complete, close the file

  printf("\nCheck the file Metrics.txt for the results!\n");

  //allocate memory for the pixels of the image thresholded at the ideal value
  imageTh=(unsigned char*)calloc(ROWS*COLS,sizeof(unsigned char));

  //generating the thresholded image for the point in the ROC curve that is closest to the top-left corner. From the MATLAB plot, this threshold value=212. No windowing
  /*thresholdIdeal=150;	      //this is the ideal value from the plot
  for(r=7;r<ROWS-7;r++)
  {
    for(c=4;c<COLS-4;c++)
    {
      if(imageMSF[(r*COLS)+c]>=thresholdIdeal)//an else statement not needed since - the idea here is to make pixels less than the theshold=0. Using 'calloc' for dynamic allocation assigns 0 to all memory
        imageTh[(r*COLS)+c]=255;	      //locations. So need to have an 'else' to do this again.
    }
  }*/

  //generating the thresholded image for the point in the ROC curve that is closest to the top-left corner. From the MATLAB plot, this threshold value=212. Using windowing
  checksum=0;
  thresholdIdeal=211;
  for(i=0;i<1262;i++)
  {
    for(r=gtr[i][1]-7;r<=gtr[i][1]+7;r++)
    {
      for(c=gtr[i][0]-4;c<=gtr[i][0]+4;c++)
      {
        if(imageMSF[(r*COLS)+c]>thresholdIdeal)
          checksum++;
      }
    }
    if(checksum>0)
      imageTh[(gtr[i][1]*COLS)+gtr[i][0]]=255;
    checksum=0;
  }
  
  //open the thresholded image file in the write format
  fpt=fopen("Thresholdimage.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the image Thresholdimage for writing\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(imageTh,ROWS*COLS,1,fpt);
  fclose(fpt);		      //writing is now complete, close the file
}
