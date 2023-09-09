#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<curses.h>

int main()
{
  system("clear");				   //clear the screen - included in curses.h

  FILE *fpt;					   //file pointer for reading the image file
  char *source="/home/rahul/Desktop/ECE 6310/Lab 3/parenthood.ppm";
  char *templ="/home/rahul/Desktop/ECE 6310/Lab 3/parenthood_e_template.ppm";
  char *gt="/home/rahul/Desktop/ECE 6310/Lab 3/parenthood_gt.txt";
  unsigned char *image,*template,*imageMSF;	   //pointer to store and access source, template, MSF and binary images
  int *zmtemplate,*imageB;			   //pointer to store and access the zero-mean template and the binary window of each character
  long int *intermediate_image;			   //pointer to store and access intermediate image pixels
  char header[320],header_T[320];		   //for reading the headers of the source image and the template
  int i,COLS,ROWS,BYTES,COLS_T,ROWS_T,BYTES_T,COLS_MSF,ROWS_MSF,BYTES_MSF,r,c,r1,c1,mean=0,index=0,j;
  long int sum=0,max=0,min=0,range;		   //for normalizing MSF image to 8 bits
  int gtr[1262][2];	    			   //row and column information of ground truth
  char gt_ch[1262];			  	   //character information of ground truth
  int count=0;					   //to save #appearances of 'e'
  int threshold;				   //for thresholding the output image of MSF
  int metric[256][4];		    		   //1st column is TP, 2nd is FP, 3rd is FN and 4th is TN
  float TPR[256],FPR[256],accuracy[256];	   //true positive rates, false positive rates and accuracies for different threshold levels
  int checksum=0;				   //to check for the presence of 'e' in the window
  bool check;					   //to check if letter is 'e' or not
  struct timespec tp1,tp2;			   //for timing the operations

  int A,B,C,D,countE2NE=0,countEneighbors=0,count1=0;   //for the conditions and #edges and #edge2ne transitions
  bool condition1,condition2,condition3,transition,flag;//boolean variables for the three codnitions and the transition
  int neighborsCW[8],neighborIndex[8],indx=0;
  int *imageB1,*imageB2;
  int edge=0,branch=0;
  int counter=0;

  int tes1,tes2;
//////////////////////////////////////////////////SOURCE IMAGE//////////////////////////////////////////////////
  //open the source image in the read format
  fpt=fopen(source,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the file parenthood.ppm for reading\n");
    exit(0);
  }

  //check if the image is in the right format (8-bit grayscale PPM)
  i=fscanf(fpt,"%s %d %d %d",header,&COLS,&ROWS,&BYTES);
  if(i!=4 || strcmp(header,"P5")!=0 || BYTES!=255)
  {
    printf("The file is not an 8-bit grayscale PPM-format image\n");
    exit(0);
  }

  //save the pixels of the source image in the memory
  image=(unsigned char*)calloc(COLS*ROWS,sizeof(unsigned char));
  header[0]=fgetc(fpt);
  fread(image,1,COLS*ROWS,fpt);
  fclose(fpt);					   //source image file is no longer needed, close it
////////////////////////////////////////////////////TEMPLATE////////////////////////////////////////////////////
  //open the source image in the read format
  fpt=fopen(templ,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the template for reading\n");
    exit(0);
  }

  //check if the template is in the right format (8-bit grayscale PPM)
  i=fscanf(fpt,"%s %d %d %d",header_T,&COLS_T,&ROWS_T,&BYTES_T);
  if(i!=4 || strcmp(header_T,"P5")!=0 || BYTES_T!=255)
  {
    printf("The file is not an 8-bit grayscale PPM-format image\n");
    exit(0);
  }

  //save the pixels of the template in the memory
  template=(unsigned char*)calloc(ROWS_T*COLS_T,sizeof(unsigned char));
  header_T[0]=fgetc(fpt);
  fread(template,1,COLS_T*ROWS_T,fpt);
  fclose(fpt);					   //template file is no longer needed, close it
//////////////////////////////////////////////////GROUND TRUTH//////////////////////////////////////////////////
  //open the ground truth file in the read format
  fpt=fopen(gt,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the groud truth file for reading\n");
    exit(0);
  }

  //save the contents of the ground truth file
  for(i=0;;i++)
  {
    fscanf(fpt,"%c %d %d\n",&gt_ch[i],&gtr[i][0],&gtr[i][1]);
    if(gt_ch[i]=='e')
      count++;
    if(gt_ch[i]=='\0')
      break;
  }
  fclose(fpt);					   //ground truth file is no longer needed, close it
  printf("The #e's (ground truth) in the source image is %d\n\n",count);
///////////////////////////////////////////////ZERO-MEAN TEMPLATE///////////////////////////////////////////////
  //allocate memory for the zero-mean template
  zmtemplate=(int *)calloc(ROWS_T*COLS_T,sizeof(int));

  //compute the mean of the zero-mean template
  for(r=0;r<ROWS_T;r++)
  {
    for(c=0;c<COLS_T;c++)
      mean+=(int)template[(r*COLS_T)+c];
  }
  mean=mean/(ROWS_T*COLS_T);
  printf("The mean value of the template pixels is %d\n",mean);

  //compute the pixel values of the zero-mean template
  for(r=0;r<ROWS_T;r++)
  {
    for(c=0;c<COLS_T;c++)
      zmtemplate[(r*COLS_T)+c]=(int)template[(r*COLS_T)+c]-mean;
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
  fclose(fpt);					   //writing is now complete, close the file
///////////////////////////////////////////////////MSF IMAGE////////////////////////////////////////////////////
  //allocate memory for the pixels of the intermediate image
  intermediate_image=(long int *)calloc(ROWS*COLS,sizeof(long int));

  //start ticking just before MSF begins
  printf("Filtering begins now...\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of filtering is %lds and %ldns\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);

  //perform cross-correlation using zero-mean template
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
  range=max-min;

  //allocate memory for the pixels of the MSF image
  imageMSF=(unsigned char *)calloc(ROWS*COLS,sizeof(unsigned char));

  //perform normalization to convert MSF image to grayscale. The formula used is 255*((pixel_value-min)/range)
  for(r=7;r<ROWS-7;r++)
  {
    for(c=4;c<COLS-4;c++)
      imageMSF[(r*COLS)+c]=(unsigned char)255*(intermediate_image[(r*COLS)+c]-min)/range;
  }

  //compute the time taken for filtering
  printf("Filtering is now complete...\n");
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of filtering is %lds and %ldns\n",(long int)tp2.tv_sec,(long int)tp2.tv_nsec);
  printf("The time taken to filter the image is %lds and %ldns\n\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)abs(tp2.tv_nsec-tp1.tv_nsec));

  //open the MSF image in the write format
  fpt=fopen("MSFimage.ppm","w");
  if(fpt==NULL)
  {
    printf("Unable to open the file MSFimage for writing\n");
    exit(0);
  }
  fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
  fwrite(imageMSF,ROWS*COLS,1,fpt);
  fclose(fpt);					   //writing is now complete, close the file
////////////////////////////////////////////////////THRESHOLDING///////////////////////////////////////////////////
  //allocate memory for the pixels of the binary image and its expanded version. The expanded version is of size 17X11 and is used for dealing with corner and edge pixels while implementing the algorithm.
  imageB=(int *)calloc(ROWS_T*COLS_T,sizeof(int));	   //to store the thresholded binary window of the original image that is centered around each ground truth location 
  imageB1=(int *)calloc((ROWS_T+2)*(COLS_T+2),sizeof(int));//expanded window (size 17X11) - zero padding along all edges makes sure the neighborhood of even pixels along the window edges are 3X3 
  imageB2=(int *)calloc((ROWS_T+2)*(COLS_T+2),sizeof(int));//copy of imageB1. Every time a pixel is marked for deletion, actual deletion happens in imageB2. Later, imageB1 updated with contents of imageB2

  //neighborindex - contents filled based on how CW neighborhood pixel inspection is carried out 
  neighborIndex[0]=0;
  neighborIndex[1]=1;
  neighborIndex[2]=2;
  neighborIndex[3]=7;
  neighborIndex[4]=3;
  neighborIndex[5]=6;
  neighborIndex[6]=5;
  neighborIndex[7]=4;

  //start ticking just before thresholding begins
  printf("Thresholding begins now...\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of filtering is %lds and %ldns\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);

  //thresholding begins here
  for(threshold=0;threshold<256;threshold++)
  {
    for(i=0;i<4;i++)		  	   //to set TP, FP, FN and TN of the current threshold value to zeros
      metric[threshold][i]=0;
    for(i=0;i<1262;i++)
    {
      if(gt_ch[i]=='e')		  	   //check if the ground truth is 'e' or not
        check=true;
      else
        check=false;
      for(r=gtr[i][1]-7;r<=gtr[i][1]+7;r++)//define window around the current ground truth location
      {
        for(c=gtr[i][0]-4;c<=gtr[i][0]+4;c++)
        {
          if(imageMSF[(r*COLS)+c]>threshold)
            checksum++;	         	  //increments every time a pixel value in the window is greater than the current threshold
          if(((int)image[(r*COLS)+c])<128)//the threshold of 128 is enough to differentiate between the black letters from the grayish background of the original image. Turn black to 1 and white to 0
            imageB[index]=1;		  //binary values are 1 and 0
          else
            imageB[index]=0;
          index++;
        }
      }
      if(checksum>0)	         	  //if checksum>0, there is at least 1 pixel with a value > the threshold. So letter detected. As letter detected, do the rest of the process inside 'if'
      {
	for(r=1;r<(ROWS_T+2-1);r++)	  //consider the inner 15X9 of the expanded array and fill it with imageB values
	{
	  for(c=1;c<(COLS_T+2-1);c++)
	  {
	    imageB1[(r*(COLS_T+2))+c]=imageB[((r-1)*COLS_T)+(c-1)];
	    imageB2[(r*(COLS_T+2))+c]=imageB[((r-1)*COLS_T)+(c-1)];
	  }
	}

	//thinning algorithm
	//do while ensures that thinning runs at least once
	do
	{
	  flag=false;			  //to check if imageB1 is the xact same as imageB2 at the end of the current round of thinning. If so, thinning terminates after flag is set.
	  for(r=1;r<(ROWS_T+2-1);r++)
	  {
	    for(c=1;c<(COLS_T+2-1);c++)
	    {
	      if(imageB1[(r*(COLS_T+2))+c]==1)//only consider white pixels 
	      {
	        indx=0;
	        countEneighbors=0;
	        countE2NE=0;
	        transition=false;
	        counter=0;
	        for(r1=-1;r1<=1;r1++)
	        {
	          for(c1=-1;c1<=1;c1++)
	          {
		    counter++;
		    if(counter!=5)
		    {
		      neighborsCW[neighborIndex[indx]]=imageB1[((r+r1)*(COLS_T+2))+(c+c1)];
		      if(neighborsCW[neighborIndex[indx]]==1)
		        countEneighbors++;
		      indx++;
		    }
	          }
	        }
	        //checking for condition 1
	        for(j=0;j<8;j++)
	        {
	          if(neighborsCW[j]==1)
		    transition=true;
	          if(neighborsCW[j]==0 && transition==true)
	          {
		    countE2NE++;
		    transition=false;
	          }
	        }
	        if(transition==true && neighborsCW[0]==0)
	          countE2NE++;
	        if(countE2NE==1)
	          condition1=true;
	        else
	          condition1=false;
	        //checking for condition 2
	        if(countEneighbors>=2 && countEneighbors<=6)
	          condition2=true;
	        else
	          condition2=false;
	        //checking for condition 3
	        A=neighborsCW[1];
	        B=neighborsCW[3];
	        C=neighborsCW[7];
	        D=neighborsCW[5];
	        if((A!=1) || (B!=1) || (C!=1 && D!=1))
	          condition3=true;
	        else
	          condition3=false;
	        //if all three conditions are satisfied, mark for deletion
	        if(condition1==true && condition2==true && condition3==true)
	          imageB2[(r*(COLS_T+2))+c]=0;
	      }
	    }
	  }	//current round of thinning is complete. Now compare imageB1 with imageB2.
	  //check if imageB1=imageB2. If so, no more thinning is possible. So flag is not set and thinning terminates.
	  for(r=0;r<(ROWS_T+2);r++)
	  {
	    for(c=0;c<(COLS_T+2);c++)
	    {
	      if(imageB1[(r*(COLS_T+2))+c]!=imageB2[(r*(COLS_T+2))+c])//even if there is one pixel that is different, set the flag
	        flag=true;
	    }
	  }
	  //replace imageB1 with imageB2 if flag is set
	  if(flag==true)
	  {
	    for(r=0;r<(ROWS_T+2);r++)
	    {
	      for(c=0;c<(COLS_T+2);c++)
	        imageB1[(r*(COLS_T+2))+c]=imageB2[(r*(COLS_T+2))+c];
	    }
	  }
	}
	while(flag==true);

/*//This part is for debugging by printing out the values
//Comment out when you do not want to debug
if(threshold==206 && i==17){
for(tes1=0;tes1<(ROWS_T+2);tes1++)
{for(tes2=0;tes2<(COLS_T+2);tes2++)
  printf("%d ",imageB1[(tes1*(COLS_T+2))+tes2]);
 printf("\n");
}
printf("\n");
for(tes1=0;tes1<ROWS_T;tes1++)
{for(tes2=0;tes2<COLS_T;tes2++)
  printf("%d ",imageB[(tes1*COLS_T)+tes2]);
 printf("\n");
}
}
///////////////////////////////////////////////////////*/
	edge=0;
	branch=0;
	for(r=1;r<(ROWS_T+2-1);r++)
	{
	  for(c=1;c<(COLS_T+2-1);c++)
	  {
	    if(imageB1[(r*(COLS_T+2))+c]==1)
	    {
	      indx=0;
	      countE2NE=0;
	      transition=false;
	      counter=0;
	      for(r1=-1;r1<=1;r1++)
	      {
	        for(c1=-1;c1<=1;c1++)
	        {
		  counter++;
		  if(counter!=5)
		  {
		    neighborsCW[neighborIndex[indx]]=imageB1[((r+r1)*(COLS_T+2))+(c+c1)];
		    indx++;
		  }
	        }
	      }
	      //check for the number of E2NE transitions
	      for(j=0;j<8;j++)
	      {
	        if(neighborsCW[j]==1)
		  transition=true;
	        if(neighborsCW[j]==0 && transition==true)
	        {
		  countE2NE++;
		  transition=false;
	        }
	      }
	      if(transition==true && neighborsCW[0]==0)
	        countE2NE++;
	      //checking if branch or edge point
	      if(countE2NE==1)
	        edge++;
	      else if(countE2NE>2)
	        branch++;
	    }
	  }
	}
	//check if the detected character is an 'e' or not
	if(edge==1 && branch==1)
	{
	  if(check==true)
	    metric[threshold][0]++;//TP
	  else
	    metric[threshold][1]++;//FP
	}
	else
	  goto nodetection;
      }
      else	       	          //if checksum=0, there is no pixel with a value greater than the threshold in the window. Consider the letter not detected (all such cases are negatives) - both FN and TN
      {
	nodetection:if(check==true)//checking if a negative is TN or not
          metric[threshold][2]++; //FN
        else
          metric[threshold][3]++; //TN
      }
      index=0;
      checksum=0;
    }
    TPR[threshold]=(float)metric[threshold][0]/(metric[threshold][0]+metric[threshold][2]);	//TPR=TP/(TP+FN)
    FPR[threshold]=(float)metric[threshold][1]/(metric[threshold][1]+metric[threshold][3]);	//FPR=FP/(FP+TN)
    accuracy[threshold]=(float)metric[threshold][0]/(metric[threshold][0]+metric[threshold][1]);//accuracy=TP/(TP+FP)
    printf("%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\n",threshold,metric[threshold][0],metric[threshold][1],metric[threshold][2],metric[threshold][3],TPR[threshold],FPR[threshold],accuracy[threshold]);
  }

  //computing the time taken for thresholding
  printf("Thresholding is complete\n");
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of thresholding is %lds and %ldns\n",(long int)tp2.tv_sec,(long int)tp2.tv_nsec);
  printf("Time taken for filtering is %lds and %ldns\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)abs(tp2.tv_nsec-tp1.tv_nsec));

  //open a file in the write format to write all metric values
  fpt=fopen("Metrics.txt","w");
  for(threshold=0;threshold<256;threshold++)
  {
    if(threshold==0)	       //to write the heading of the file
      fprintf(fpt,"Thr\tTP\tFP\tFN\tTN\tTPR\tFPR\tAcc\n");
    fprintf(fpt,"%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\n",threshold,metric[threshold][0],metric[threshold][1],metric[threshold][2],metric[threshold][3],TPR[threshold],FPR[threshold],accuracy[threshold]);
  }
  fclose(fpt);		      //writing is now complete, close the file

  printf("\nCheck the file Metrics.txt for the results!\n\n");
}
