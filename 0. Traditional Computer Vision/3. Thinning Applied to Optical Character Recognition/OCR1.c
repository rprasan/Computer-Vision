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
  int *zmtemplate,*imageB,*imageB1,*mark; 	   //pointer to store and access the zero-mean template, the binary window, the expanded binary window and the indices marked for deletion
  long int *intermediate_image;			   //pointer to store and access intermediate image pixels
  char header[320],header_T[320];		   //for reading the headers of the source image and the template
  int i,COLS,ROWS,BYTES,COLS_T,ROWS_T,BYTES_T,COLS_MSF,ROWS_MSF,BYTES_MSF,r,c,r1,c1,r2,c2,mean=0,edgeFlag,branchFlag,indxThin,j,k;
  long int sum=0,max=0,min=0,range;		   //for normalizing MSF image to 8 bits
  int gtr[1262][2];	    			   //row and column information of ground truth
  char gt_ch[1262];			  	   //character information of ground truth
  int count=0;					   //to save #appearances of 'e'
  int threshold;				   //for thresholding the output image of MSF
  int metric[256][4];		    		   //1st column is TP, 2nd is FP, 3rd is FN and 4th is TN
  float TPR[256],FPR[256],accuracy[256];	   //true positive rates, false positive rates and accuracies for different threshold levels
  unsigned char neighborsCW[10];		   //for the neighborhood and clockwise operation
  bool flag,check;				   //if detected, FLAG=TRUE. If ground truth is 'e', check=TRUE.
  int checkThin=0;				   //to check for the presence of marked pixels
  int e2neC1,eC2;				   //for the first two of the three conditions for marking a pixel for deletion
  bool transition;			   	   //for marking a pixel for deletion and flag for E2NE transitions
  int edge,branch;				   //counter for edge and branch points
  int A,B,C,D;					   //for condition two of mearking for deletion
  int neighborIndex[9];				   //to store the neighborhood
  int counter=0,indx=0;				   //for generating the neighborhood
  struct timespec tp1,tp2;			   //for timing the operations

int tes1,tes2,*imageB2;
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
  imageB=(int *)calloc(ROWS*COLS,sizeof(int));		  //original window
  imageB1=(int *)calloc((ROWS_T+2)*(COLS_T+2),sizeof(int));//expanded window-with zero padding around edges
  imageB2=(int *)calloc((ROWS_T+2)*(COLS_T+2),sizeof(int));

  //for storing the indices to be marked for deletion during each round of iteration
  mark=(int *)calloc(ROWS_T*COLS_T,sizeof(int));

  //neighborindex
  neighborIndex[0]=6;
  neighborIndex[1]=7;
  neighborIndex[2]=8;
  neighborIndex[3]=5;
  neighborIndex[4]=1;
  neighborIndex[5]=4;
  neighborIndex[6]=3;
  neighborIndex[7]=2;
  neighborIndex[8]=9;

  //start ticking just before thresholding begins
  printf("Thresholding begins now...\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of filtering is %lds and %ldns\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);

  //the threshold of 128 is enough to differentiate between the black letters from the grayish background of the original image
  for(r=7;r<(ROWS-7);r++)
  {
    for(c=4;c<(COLS-4);c++)
    {
      if((int)image[(r*COLS)+c]>128)
	imageB[(r*COLS)+c]=1;
      else
	imageB[(r*COLS)+c]=0;
    }
  }

  //thresholding begins here
  for(threshold=0;threshold<256;threshold++)
  {
    for(i=0;i<4;i++)		  //to set TP, FP, FN and TN of the current threshold value to zeros
      metric[threshold][i]=0;
    for(i=0;i<1262;i++)
    {
      //check if ground truth is 'e' or not
      check=false;
      if(gt_ch[i]=='e')
	check=true;
      //to check if the character is detected from the MSF image
      flag=false;
      for(r1=gtr[i][1]-7;r1<=gtr[i][1]+7;r1++)
      {
        for(c1=gtr[i][0]-4;c1<=gtr[i][0]+4;c1++)
        {
	  if(imageMSF[(r1*COLS)+c1]>threshold)
	    flag=true;		  //character detected in the window
        }
      }
      if(flag==true)		  //need to perform thinning only if detected
      {	
	for(r=0;r<((ROWS_T+2)*(COLS_T+2));r++)
	  imageB1[(r*(COLS_T+2))+c]=1;
	r=1;
	for(r1=gtr[i][1]-7;r1<=gtr[i][1]+7;r1++)
	{
	  c=1;
	  for(c1=gtr[i][0]-4;c1<=gtr[i][0]+4;c1++)
	  {
	    imageB1[(r*(COLS_T+2))+c]=imageB[(r1*COLS)+c1];
	    c+=1;
	  }
	  r+=1;
	}

	//to check for the three conditions
	e2neC1=0;
	eC2=0;
	//use do while so that the thinning algorithm runs at least once
	do
	{
	  checkThin=0;//to check if there are any pixels marked for deletion. If not, checkThin will be zero at the end of the current round of thinning.
	  c2=0;
	  edge=0;
	  branch=0;
	  for(r=1;r<(ROWS_T+2-1);r++)
	  {
	    for(c=1;c<(COLS_T+2-1);c++)
	    {
	      if(imageB1[(r*(COLS_T+2))+c]==0)
	      {
		counter=0;
		indx=0;
		for(j=-1;j<=1;j++)
		{
		  for(k=-1;k<=1;k++)
		  {
		    counter++;
		    if(counter!=5)
		    {
		      neighborsCW[neighborIndex[indx]]=imageB1[((r+j)*(COLS_T+2))+(c+k)];
		      indx++;
		    }
		  }
		}
		neighborsCW[neighborIndex[indx]]=imageB1[(r*(COLS_T+2))+(c+1)];
		A=neighborsCW[7];
		B=neighborsCW[1];
		C=neighborsCW[5];
		D=neighborsCW[3];
	      }
	      transition=false;
	      e2neC1=0;
	      edgeFlag=0;
	      branchFlag=0;
	      //check for condition 1
	      for(r2=1;r2<(COLS_T+2-1);r2++)
	      {
		if(neighborsCW[r2]==0)
		  transition=true;
		else if(neighborsCW[r2]==1 && transition==true)
		{
		  e2neC1++;	//transition detected
		  transition=false;
		}
		else
		  transition=false;
              }
	      //exactly one E2NE transition for edge point
	      if(e2neC1==1 && edgeFlag==0)
	      {
		edge=1;
		edgeFlag=1;
	      }	
	      else if(e2neC1==1 && edgeFlag==1)
		edge=0;
	      //greater than 2 E2NE transitions for branch point
	      else if(e2neC1==2 && branchFlag==0)
	      {
		branch=1;
		branchFlag=1;
	      }
	      else if(e2neC1==2 && branchFlag==1)
		branch=0;
	      else if(e2neC1>2)
	        branch=0;
	      eC2=0;
	      if(e2neC1==1)
	      {
		for(r2=1;r2<COLS_T;r2++)
		{
		  if(neighborsCW[r2]==0)
		    eC2++;
		}
	      }
	      //check if all three conditions are met. If so, mark for deletion.
	      if(e2neC1==1 && (eC2>=3 && eC2<=7) && (A==1 || B==1 || (C==1 && D==1)))
	      {
		mark[c2]=(r*(COLS_T+2))+c;
		c2++;
		checkThin++;
	      }
	    }
	  }
	  if(checkThin>0)
	  {
	    for(c2=0;c2<checkThin;c2++)
	    {
	      indxThin=mark[c2];
	      imageB1[indxThin]=1;
	    }
	  }
	}
	while(checkThin>0);//if checkThin=0, no more pixels are marked for deletion. Terminate thinning for the current ground truth entry here.



////////////////////////////////////////////////////
//This bounded part is used to generate the thinned
//for the optimum threshold value of 206 for a ground
//location that corresponds to an 'e'(location 3 is e)
//Don't use this part during normal coding.
if(threshold==206 && i==2)
{
  for(tes1=0;tes1<(ROWS_T+2);tes1++)
  {
    printf("\t\t\t\t\t\t\t\t\t\t");
    for(tes2=0;tes2<(COLS_T+2);tes2++)
    {
      if((int)imageB1[(tes1*(COLS_T+2))+tes2]==0)
	imageB2[(tes1*(COLS_T+2))+tes2]=255;
      else
	imageB2[(int)(tes1*(COLS_T+2))+tes2]=0;
      printf("%d ",imageB1[(tes1*(COLS_T+2))+tes2]);
    }
    printf("\n");
  }
  printf("\n\n\n");
}

//printf("\n\n");
if(threshold==206 && i==2)
{
  for(tes1=0;tes1<(ROWS_T+2);tes1++)
  {
    printf("\t\t\t\t\t\t\t");
    for(tes2=0;tes2<(COLS_T+2);tes2++)
      printf("%d\t",imageB2[(tes1*(COLS_T+2))+tes2]);
    printf("\n");
  }
  printf("\n");
}
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


	//check if the detected character is an 'e' or not		
	if(edge==1 && branch==1)
	{
	  if(check==true)
	    metric[threshold][0]++;
	  else
	    metric[threshold][1]++;
	}
	else
	  goto repeat;
      }
      else
      {
	repeat:if(check==true)
	  metric[threshold][2]++;
	else
	  metric[threshold][3]++;
      }
    }
    TPR[threshold]=(float)metric[threshold][0]/(metric[threshold][0]+metric[threshold][2]);	//TPR=TP/(TP+FN)
    FPR[threshold]=(float)metric[threshold][1]/(metric[threshold][1]+metric[threshold][3]);	//FPR=FP/(FP+TN)
    accuracy[threshold]=(float)metric[threshold][0]/(metric[threshold][0]+metric[threshold][1]);//accuracy=TP/(TP+FP)
    //printf("%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\n",metric[threshold][0],metric[threshold][1],metric[threshold][2],metric[threshold][3],TPR[threshold],FPR[threshold],accuracy[threshold]);
  }
  
  //computing the time taken for thresholding
  printf("Thresholding is complete\n");
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Time at the end of thresholding is %lds and %ldns\n",(long int)tp2.tv_sec,(long int)tp2.tv_nsec);
  printf("Time taken for filtering is %lds and %ldns\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)abs(tp2.tv_nsec-tp1.tv_nsec));

  //open a file in the write format to write all metric values
  /*fpt=fopen("Metrics.txt","w");
  for(threshold=0;threshold<256;threshold++)
  {
    if(threshold==0)	       //to write the heading of the file
      fprintf(fpt,"TP\tFP\tFN\tTN\tTPR\tFPR\tAcc\n");
    fprintf(fpt,"%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\n",metric[threshold][0],metric[threshold][1],metric[threshold][2],metric[threshold][3],TPR[threshold],FPR[threshold],accuracy[threshold]);
  }
  fclose(fpt);		      //writing is now complete, close the file

  printf("\nCheck the file Metrics.txt for the results!\n");*/

}
