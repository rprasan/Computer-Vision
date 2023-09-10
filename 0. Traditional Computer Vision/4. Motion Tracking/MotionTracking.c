#include<time.h>
#include<math.h>
#include<stdio.h>
#include<string.h>
#include<curses.h>
#include<stdlib.h>

int main()
{
  system("clear");

  FILE *fpt;					       //file pointer for reading the source file
  char *source="/home/rahul/Desktop/ECE 6310/Lab 7/acc_gyro.txt";
  int i,j,n;
  float data[1250][7];				       //array to store the data - the format is time, Xacc, Yacc, Zacc, pitch, roll and yaw. Use float as the data contains fractions
  int flag,subflag;				       //for various flag operations related to extracting a window of samples from the dataset
  int windowXacc=10,windowYacc=10,windowZacc=10;       //for storing the window of accelerations
  int windowPang=10,windowRang=10,windowYang=10;       //for storing the window of angle changes
  int *indicesXacc,*indicesYacc,*indicesZacc;          //to store the indices of the dataset that correspond to the time at which the object is in motion
  int *indicesPang,*indicesRang,*indicesYang;          //to store the indices of the dataset that correspond to the time at which the object is experiencing a change in orientation
  float varThXacc,varThYacc,varThZacc;		       //variance thresholds for identifying the region in the dataset where there is motion
  float varThPang,varThRang,varThYang;		       //variance thresholds for identifying the region in the dataset where there is a change in orientation
  float testArrayXacc[windowXacc],testArrayYacc[windowYacc],testArrayZacc[windowZacc];
  float testArrayPang[windowPang],testArrayRang[windowRang],testArrayYang[windowYang];
  float sum,mean,variance;			       //used to compute the variance of the window of samples
  int nmX=0,nmY=0,nmZ=0;			       //nmX, nmY, nmZ - #motions along the three axes - to remove noisy area
  int nmP=0,nmR=0,nmYa=0;			       //nmP, nmR, nmYa - #regions in pitch, roll and yaw data - noisy regions must be removed
  float meanX=0,varX=0,meanY=0,varY=0,meanZ=0,varZ=0;  //meanX, meanY, meanZ - avg of respective acceleration, varX, varY, varZ - variance of respective acceleration
  float meanP=0,varP=0,meanR=0,varR=0,meanYa=0,varYa=0;//meanP, meanR, meanYa - avg of respective orientation, varP, varR, varYa - variance of respective orientation
  float sampTime=0.05,velPrevious=0,velCurrent=0,velFinal=0,velAverage=0;//to calculate the displacement
  struct timespec tp1,tp2;			       //for timing the operations

//////////////////////////////////////////////////////DATA//////////////////////////////////////////////////////
  //open the source file for reading
  fpt=fopen(source,"rb");
  if(fpt==NULL)
  {
    printf("Unable to open the file %s for reading!\n",source);
    exit(0);
  }

  //save the data samples in an array and print it - the unit of all three accelerations is G and the unit of all angles is Radians per second
  i=0;
  while(fscanf(fpt,"%f %f %f %f %f %f %f\n",&data[i][0],&data[i][1],&data[i][2],&data[i][3],&data[i][4],&data[i][5],&data[i][6])==7) //#items should be 7
  {
    if(i==0)
    {
      printf("\nThe data samples are:\n");
      printf("________________________________________________________________________________________________________\n");
      printf("TIME\t\t Xacc\t\t Yacc\t\t  Zacc\t\t PITCH\t\t ROLL\t\t  YAW\n____________________________________________________________________________________________________");
      printf("____\n");
    }
    printf("%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n",data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6]);
    i++;
  }
  fclose(fpt);					       //file no longer needed, close the pointer
  n=i;
  printf("\nThe total number of data samples is %d\n\n",n);
  printf("---------------------------------------------------------------------------------------------------------------------------------\n\n");

  //convert all accelerations from G to m/s^2
  for(i=0;i<n;i++)
  {
    data[i][1]=9.81*data[i][1];
    data[i][2]=9.81*data[i][2];
    data[i][3]=9.81*data[i][3];
  }

////////////////////////////////////////////////////WINDOWING///////////////////////////////////////////////////
  //variables to store indices corresponding to object motion and change in angle
  indicesXacc=(int*)calloc(n,sizeof(int));	       //store indices of the samples corresponding to the occurence of motion. As calloc is used, only when there is motion, the corresponding entries in 
  indicesYacc=(int*)calloc(n,sizeof(int));	       //this array are non-zeros. We are not interested  in the rest of the indices since the object is stationary at other times.
  indicesZacc=(int*)calloc(n,sizeof(int));
  indicesPang=(int*)calloc(n,sizeof(int));
  indicesRang=(int*)calloc(n,sizeof(int));
  indicesYang=(int*)calloc(n,sizeof(int));

  //start ticking just before windowing begins
  printf("Windowing begins now...\n\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of windowing is %lds and %ldns.\n\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);

  //detecting duration of motion alogn the x-axis
  varThXacc=0.4;				       //this is the variance threshold for acceleration along the x-axis
  flag=0;
  subflag=0;
  for(i=0;i<(n-windowXacc);i++)			       //this loop slides the window across the length of the entire dataset
  {
    sum=0;
    mean=0;
    variance=0;
    for(j=0;j<windowXacc;j++)			       //this loop generates the window of samples for he current iteration
    {
      testArrayXacc[j]=data[i+j][1];
      sum+=testArrayXacc[j];
    }
    mean=sum/windowXacc;			       //compute the mean of the window of samples
    for(j=0;j<windowXacc;j++)
      variance+=(testArrayXacc[j]-mean)*(testArrayXacc[j]-mean);//you need the sum of the squares of the difference between each sample and the mean in order to calculate the variance
    variance=variance/windowXacc;		       //compute the variance
    //one motion sequence in the dataset is divided into three regions - flag 1 represents the region between the first time instance at which the variance of the window crossed the threshold and flag 2,
    //flag 2 indicates the duration of motion during which acceleration largely remains constant (the table-top region - variance is low here), and flag 3 represents the region between the time instant
    //at which the variance of the window crosses the threshold due to the object decelerating and the time instant at which acceleration becomes nearly zero
    if(flag==0 && variance>varThXacc)		       //the first region begins here
      flag=1;
    else if(flag==1 && variance>varThXacc)	       //as the objects keeps accelerating, the variance is still greater than the threshold. So, we are still in region 1.
    {
      flag=1;
      subflag=1;
    }
    else if(flag==1 && variance<varThXacc)	       //we approach the table-top region which is region 2
      flag=2;
    else if(flag==2 && variance>varThXacc)	       //as the object decelerates, the variance again becomes greater than the threshold. So we move into region 3.
      flag=3;
    else if(flag==3 && variance>varThXacc)	       //as the objects keeps decelerating, the variance is still greater than the threshold. So, we are still in region 3.
      flag=3;
    else if(flag==3 && variance<varThXacc)	       //region 3 terminates when the object becomes stationary again. So, we set flag to 0.
    {
      flag=0;
      subflag=0;				       //if subflag is not reset to 0 and there are multiple table-top regions (multiple cases of motion), only the first one will be properly detected
    }
//printf("%d %d %d\n",i,flag,subflag);
    //we extract the indices based on the region that we are in at a given window instant    
    if(flag!=0)
    {
      if(flag==1 && subflag==0)
      {
        for(j=0;j<windowXacc;j++)
	{
          //printf("%d %f\n",i+j,testArrayXacc[j]);
	  indicesXacc[i+j]=i+j;			       //storing the indices
	}
      }
      else
      {
        //printf("%d %f\n",i+windowXacc-1,testArrayXacc[windowXacc-1]);
	indicesXacc[i+(windowXacc-1)]=i+(windowXacc-1);//store the index - -1 is necessary as array indexing starts from 0 and not 1
      }
    }
  }
/*
  //eliminate the noisy region from indicesXacc
  flag=0;
  for(i=0;i<n;i++)
  { 
    if(indicesXacc[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmX++;
    }
    if(indicesXacc[i]==0)			       //when indicesXacc[i] becomes zero again (when current region ends), reset the flag
      flag=0;
//printf("%d %d %d\n",i,flag,nmX);
  }
  int regionLengthX[nmX],currentLengthX=0,start=0;     //regionLengthX - store each region's length, currentLengthX - length of the region that 'i' is currently in, start - starting point of current region
  for(i=0;i<nmX;i++)
    regionLengthX[i]=0;				       //initialize length of each region to 0
  flag=0;
  for(i=0;i<n;i++)
  {
    if(indicesXacc[i]!=0 && flag==0)		       //check if beginning of a region - this condition is true only at the beginning of each region (first index of a region)
    {
      flag=1;					       //set flag as region entered
      start=i;
      meanX+=data[i][1];
      currentLengthX++;
    }
    else if(indicesXacc[i]!=0 && flag==1)	       //check if 'i' is still in the region - this condition is true as long as 'i' is inside a region
    {
      meanX+=data[i][1];
      currentLengthX++;      
    }
    else if(indicesXacc[i]==0 && flag==1)	       //check if end of the current region is reached
    {
      meanX=meanX/currentLengthX;		       //compute the mean
      for(j=start;j<i;j++)			       //compute the variance of all elements of the last region
	varX+=(data[j][1]-meanX)*(data[j][1]-meanX);
      varX=varX/currentLengthX;
//printf("\n\n%d %d %f",start,i-1,varX);	       //uncomment to get the variance of each region and use that information to set the threshold in the next statement
      if(varX<5)				       //check if variance of last region<threshold or not. If <threshold, replace contents of indices corresponding to the region in indicesYacc with all 0s
	for(j=start;j<i;j++)			       //This eliminates the first region which is, in fact, just noise
	  indicesXacc[j]=0;
      currentLengthX=0;				       //set all these to zeros so that they can be used again for the next region
      meanX=0;
      varX=0;
      flag=0;
    }
  }
*/
  //detecting duration of motion alogn the y-axis
  varThYacc=0.50;				       //this is the variance threshold for acceleration along the y-axis
  flag=0;
  subflag=0;
  for(i=0;i<(n-windowYacc);i++)			       //this loop slides the window across the length of the entire dataset
  {
    sum=0;
    mean=0;
    variance=0;
    for(j=0;j<windowYacc;j++)			       //this loop generates the window of samples for he current iteration
    {
      testArrayYacc[j]=data[i+j][2];
      sum+=testArrayYacc[j];
    }
    mean=sum/windowYacc;			       //compute the mean of the window of samples
    for(j=0;j<windowYacc;j++)
      variance+=(testArrayYacc[j]-mean)*(testArrayYacc[j]-mean);//you need the sum of the squares of the difference between each sample and the mean in order to calculate the variance
    variance=variance/windowYacc;		       //compute the variance
    //same explanation as motion detection along the x-axis
    if(flag==0 && variance>varThYacc)		       //the first region begins here
      flag=1;
    else if(flag==1 && variance>varThYacc)	       //as the objects keeps accelerating, the variance is still greater than the threshold. So, we are still in region 1.
    {
      flag=1;
      subflag=1;
    }
    else if(flag==1 && variance<varThYacc)	       //we approach the table-top region which is region 2
      flag=2;
    else if(flag==2 && variance>varThYacc)	       //as the object decelerates, the variance again becomes greater than the threshold. So we move into region 3.
      flag=3;
    else if(flag==3 && variance>varThYacc)	       //as the objects keeps decelerating, the variance is still greater than the threshold. So, we are still in region 3.
      flag=3;
    else if(flag==3 && variance<varThYacc)	       //region 3 terminates when the object becomes stationary again. So, we set flag to 0.
    {
      flag=0;
      subflag=0;				       //if subflag is not reset to 0 and there are multiple table-top regions (multiple cases of motion), only the first one will be properly detected
    }
//printf("%d %d %d\n",i,flag,subflag);
    //we extract the indices based on the region that we are in at a given window instant        
    if(flag!=0)
    {
      if(flag==1 && subflag==0)
      {
        for(j=0;j<windowYacc;j++)
	{
          //printf("%d %f\n",i+j,testArrayYacc[j]);
	  indicesYacc[i+j]=i+j;			       //storing the indices
	}
      }
      else
      {
        //printf("%d %f\n",i+windowYacc-1,testArrayYacc[windowYacc-1]);
	indicesYacc[i+(windowYacc-1)]=i+(windowYacc-1);//store the index - -1 is necessary as array indexing starts from 0 and not 1
      }
    }
  }
/*
  //eliminate the noisy region from indicesYacc
  flag=0;
  for(i=0;i<n;i++)
  { 
    if(indicesYacc[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmY++;
    }
    if(indicesYacc[i]==0)			       //when indicesYacc[i] becomes zero again (when current region ends), reset the flag
      flag=0;
//printf("%d %d %d\n",i,flag,nmY);
  }
  int regionLengthY[nmY],currentLengthY=0;	       //regionLengthY - store length of each region, currentLengthY - length of the region that 'i' is currently in
  start=0;					       //start - starting point of current region
  for(i=0;i<nmY;i++)
    regionLengthY[i]=0;				       //initialize length of each region to 0
  flag=0;
  for(i=0;i<n;i++)
  {
    if(indicesYacc[i]!=0 && flag==0)		       //check if beginning of a region - this condition is true only at the beginning of each region (first index of a region)
    {
      flag=1;					       //set flag as region entered
      start=i;
      meanY+=data[i][2];
      currentLengthY++;
    }
    else if(indicesYacc[i]!=0 && flag==1)	       //check if 'i' is still in the region - this condition is true as long as 'i' is inside a region
    {
      meanY+=data[i][2];
      currentLengthY++;      
    }
    else if(indicesYacc[i]==0 && flag==1)	       //check if end of the current region is reached
    {
      meanY=meanY/currentLengthY;		       //compute the mean
      for(j=start;j<i;j++)			       //compute the variance of all elements of the last region
	varY+=(data[j][2]-meanY)*(data[j][2]-meanY);
      varY=varY/currentLengthY;
//printf("\n\n%d %d %f",start,i-1,varY);	       //uncomment to get the variance of each region and use that information to set the threshold in the next statement
      if(varY<5)				       //check if variance of last region<threshold or not. If <threshold, replace contents of indices corresponding to the region in indicesYacc with all 0s
	for(j=start;j<i;j++)			       //This eliminates the first region which is, in fact, just noise
	  indicesYacc[j]=0;
      currentLengthY=0;				       //set all these to zeros so that they can be used again for the next region
      meanY=0;
      varY=0;
      flag=0;
    }
  }
*/
  //detecting duration of motion alogn the z-axis
  varThZacc=0.55;				       //this is the variance threshold for acceleration along the z-axis
  flag=0;
  subflag=0;
  for(i=0;i<(n-windowZacc);i++)			       //this loop slides the window across the length of the entire dataset
  {
    sum=0;
    mean=0;
    variance=0;
    for(j=0;j<windowZacc;j++)			       //this loop generates the window of samples for the current iteration
    {
      testArrayZacc[j]=data[i+j][3];
      sum+=testArrayZacc[j];
    }
    mean=sum/windowZacc;			       //compute the mean of the window of samples
    for(j=0;j<windowZacc;j++)
      variance+=(testArrayZacc[j]-mean)*(testArrayZacc[j]-mean);//you need the sum of the squares of the difference between each sample and the mean in order to calculate the variance
    variance=variance/windowZacc;		       //compute the variance
    //same explanation as motion detection along the x-axis
    if(flag==0 && variance>varThZacc)		       //the first region begins here
      flag=1;
    else if(flag==1 && variance>varThZacc)	       //as the objects keeps accelerating, the variance is still greater than the threshold. So, we are still in region 1.
    {
      flag=1;
      subflag=1;
    }
    else if(flag==1 && variance<varThZacc)	       //we approach the table-top region which is region 2
      flag=2;
    else if(flag==2 && variance>varThZacc)	       //as the object decelerates, the variance again becomes greater than the threshold. So we move into region 3.
      flag=3;
    else if(flag==3 && variance>varThZacc)	       //as the objects keeps decelerating, the variance is still greater than the threshold. So, we are still in region 3.
      flag=3;
    else if(flag==3 && variance<varThZacc)	       //region 3 terminates when the object becomes stationary again. So, we set flag to 0.
    {
      flag=0;
      subflag=0;				       //if subflag not reset to 0 and there are multiple table-top regions (multiple cases of motion (z-axis has it)), only the first one is properly detected
    }
//printf("%d %d %d\n",i,flag,subflag);
    //we extract the indices based on the region that we are in at a given window instant    
    if(flag!=0)
    {
      if(flag==1 && subflag==0)
      {
        for(j=0;j<windowZacc;j++)
	{
          //printf("%d %f\n",i+j,testArrayZacc[j]);
	  indicesZacc[i+j]=i+j;			       //storing the indices
	}
      }
      else
      {
        //printf("%d %f\n",i+windowZacc-1,testArrayZacc[windowZacc-1]);
	indicesZacc[i+(windowZacc-1)]=i+(windowZacc-1);//store the index - -1 is necessary as array indexing starts from 0 and not 1
      }
    }
  }
/*
  //eliminate the noisy region from indicesZacc
  flag=0;
  for(i=0;i<n;i++)
  { 
    if(indicesZacc[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmZ++;
    }
    if(indicesZacc[i]==0)			       //when indicesZacc[i] becomes zero again (when current region ends), reset the flag
      flag=0;
//printf("%d %d %d\n",i,flag,nmZ);
  }
  int regionLengthZ[nmZ],currentLengthZ=0;	       //regionLengthZ - store length of each region, currentLengthZ - length of the region that 'i' is currently in
  start=0;					       //start - starting point of current region
  for(i=0;i<nmZ;i++)
    regionLengthZ[i]=0;				       //initialize length of each region to 0
  flag=0;
  for(i=0;i<n;i++)
  {
    if(indicesZacc[i]!=0 && flag==0)		       //check if beginning of a region - this condition is true only at the beginning of each region (first index of a region)
    {
      flag=1;					       //set flag as region entered
      start=i;
      meanZ+=data[i][3];
      currentLengthZ++;
    }
    else if(indicesZacc[i]!=0 && flag==1)	       //check if 'i' is still in the region - this condition is true as long as 'i' is inside a region
    {
      meanZ+=data[i][3];
      currentLengthZ++;      
    }
    else if(indicesZacc[i]==0 && flag==1)	       //check if end of the current region is reached
    {
      meanZ=meanZ/currentLengthZ;		       //compute the mean
      for(j=start;j<i;j++)			       //compute the variance of all elements of the last region
	varZ+=(data[j][3]-meanZ)*(data[j][3]-meanZ);
      varZ=varZ/currentLengthZ;
//printf("\n\n%d %d %f",start,i-1,varZ);	       //uncomment to get the variance of each region and use that information to set the threshold in the next statement
      if(varZ<7)				       //check if variance of last region<threshold or not. If <threshold, replace contents of indices corresponding to the region in indicesZacc with all 0s
	for(j=start;j<i;j++)			       //This eliminates the first region which is, in fact, just noise
	  indicesZacc[j]=0;
      currentLengthZ=0;				       //set all these to zeros so that they can be used again for the next region
      meanZ=0;
      varZ=0;
      flag=0;
    }
  }
*/
  //detecting the duration of change in pitch
  varThPang=0.015;				       //this is the variance threshold for change in pitch
  flag=0;					       //the flag is set when a new region is identified
  subflag=0;					       //the subflag is set when control is inside the new region
  float interMean=0;				       //to find the mean of a window of samples inside the body of the current region
  float abso;	 				       //used to find the absolute value as the function abs() does not work with floating point numbers
  for(i=0;i<(n-windowPang);i++)			       //this loop slides the window across the length of the entire dataset
  {
    sum=mean=variance=0;
    for(j=0;j<windowPang;j++)			       //this loop generates the window of samples for the current iteration
    {
      testArrayPang[j]=data[i+j][4];
      sum+=testArrayPang[j];
    }
    mean=sum/windowPang;			       //compute the mean of the window of samples
    for(j=0;j<windowPang;j++)
      variance+=(testArrayPang[j]-mean)*(testArrayPang[j]-mean);//you need the sum of the squares of the difference between each sample and the mean in order to calculate the variance
    variance=variance/windowPang;		       //compute the variance
    if(variance>varThPang && flag==0)		       //check if it is the beginning of a new region
    {
      flag=1;					       //set the flag as a new region has started
      subflag=1;				       //set the subflag as control must now go inside the region's body
    }
    if(subflag==1)
    {
      indicesPang[i]=i;
      interMean=0;
      for(j=0;j<windowPang;j++)
	interMean+=data[i+j][4];
      interMean=interMean/windowPang;
      if(interMean<0)				       //finding the absolute value of the intermediate mean - the strategy is to find the end point based on when exactly the mean of the chosen window goes
	abso=(-1)*interMean;			       //below a certain threshold. As the dataset has pitch variations that are both positive and negative, we must use the magnitude for thresholding.
      else
	abso=interMean;
      if(abso<0.001824)				       //use mean thresholding to identify if the end of the region has been reached. If the end is reached, reset the flags.
      {
	subflag=0;
	flag=0;
      }
    }
    //printf("%d %d %f\n",i,indicesPang[i],data[i][4]);
  }
/*
  //eliminate the noisy region from indicesPang
  flag=0;
  for(i=0;i<n;i++)
  { 
    if(indicesPang[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmP++;
    }
    if(indicesPang[i]==0)			       //when indicesPang[i] becomes zero again (when current region ends), reset the flag
      flag=0;
//printf("%d %d %d\n",i,flag,nmP);
  }
  int regionLengthP[nmP],currentLengthP=0; 	       //regionLengthP - store length of each region, currentLengthP - length of the region that 'i' is currently in
  start=0;					       //start - starting point of current region
  for(i=0;i<nmP;i++)
    regionLengthP[i]=0;				       //initialize length of each region to 0
  flag=0;
  for(i=0;i<n;i++)
  {
    if(indicesPang[i]!=0 && flag==0)		       //check if beginning of a region - this condition is true only at the beginning of each region (first index of a region)
    {
      flag=1;					       //set flag as region entered
      start=i;
      meanP+=data[i][4];
      currentLengthP++;
    }
    else if(indicesPang[i]!=0 && flag==1)	       //check if 'i' is still in the region - this condition is true as long as 'i' is inside a region
    {
      meanP+=data[i][4];
      currentLengthP++;      
    }
    else if(indicesPang[i]==0 && flag==1)	       //check if end of the current region is reached
    {
      meanP=meanP/currentLengthP;		       //compute the mean
      for(j=start;j<i;j++)			       //compute the variance of all elements of the last region
	varP+=(data[j][4]-meanP)*(data[j][4]-meanP);
      varP=varP/currentLengthP;
//printf("\n\n%d %d %f",start,i-1,varP);	       //uncomment to get the variance of each region and use that information to set the threshold in the next statement
      if(varP<0.1)				       //check if variance of last region<threshold or not. If <threshold, replace contents of indices corresponding to the region in indicesPang with all 0s
	for(j=start;j<i;j++)			       //This eliminates the smaller regions which is, in fact, just noise
	  indicesPang[j]=0;
      currentLengthP=0;				       //set all these to zeros so that they can be used again for the next region
      meanP=0;
      varP=0;
      flag=0;
    }
  }
*/
  //detecting the duration of change in roll
  varThRang=0.015;				       //this is the variance threshold for change in roll
  flag=0;					       //the flag is set when a new region is identified
  subflag=0;					       //the subflag is set when control is inside the new region
  interMean=0;					       //to find the mean of a window of samples inside the body of the current region
  abso;	 				 	       //used to find the absolute value as the function abs() does not work with floating point numbers
  for(i=0;i<(n-windowRang);i++)			       //this loop slides the window across the length of the entire dataset
  {
    sum=mean=variance=0;
    for(j=0;j<windowRang;j++)			       //this loop generates the window of samples for the current iteration
    {
      testArrayRang[j]=data[i+j][5];
      sum+=testArrayRang[j];
    }
    mean=sum/windowRang;			       //compute the mean of the window of samples
    for(j=0;j<windowRang;j++)
      variance+=(testArrayRang[j]-mean)*(testArrayRang[j]-mean);//you need the sum of the squares of the difference between each sample and the mean in order to calculate the variance
    variance=variance/windowRang;		       //compute the variance
    if(variance>varThRang && flag==0)		       //check if it is the beginning of a new region
    {
      flag=1;					       //set the flag as a new region has started
      subflag=1;				       //set the subflag as control must now go inside the region's body
    }
    if(subflag==1)
    {
      indicesRang[i]=i;
      interMean=0;
      for(j=0;j<windowRang;j++)
	interMean+=data[i+j][5];
      interMean=interMean/windowRang;
      if(interMean<0)				       //finding the absolute value of the intermediate mean - the strategy is to find the end point based on when exactly the mean of the chosen window goes
	abso=(-1)*interMean;			       //below a certain threshold. As the dataset has pitch variations that are both positive and negative, we must use the magnitude for thresholding.
      else
	abso=interMean;
      if(abso<0.001824)				       //use mean thresholding to identify if the end of the region has been reached. If the end is reached, reset the flags.
      {
	subflag=0;
	flag=0;
      }
    }
    //printf("%d %d %f\n",i,indicesPang[i],data[i][5]);
  }
/*
  //eliminate the noisy region from indicesRang
  flag=0;
  for(i=0;i<n;i++)
  { 
    if(indicesRang[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmR++;
    }
    if(indicesRang[i]==0)			       //when indicesRang[i] becomes zero again (when current region ends), reset the flag
      flag=0;
//printf("%d %d %d\n",i,flag,nmR);
  }
  int regionLengthR[nmR],currentLengthR=0; 	       //regionLengthR - store length of each region, currentLengthR - length of the region that 'i' is currently in
  start=0;					       //start - starting point of current region
  for(i=0;i<nmR;i++)
    regionLengthR[i]=0;				       //initialize length of each region to 0
  flag=0;
  for(i=0;i<n;i++)
  {
    if(indicesRang[i]!=0 && flag==0)		       //check if beginning of a region - this condition is true only at the beginning of each region (first index of a region)
    {
      flag=1;					       //set flag as region entered
      start=i;
      meanR+=data[i][5];
      currentLengthR++;
    }
    else if(indicesRang[i]!=0 && flag==1)	       //check if 'i' is still in the region - this condition is true as long as 'i' is inside a region
    {
      meanR+=data[i][5];
      currentLengthR++;      
    }
    else if(indicesRang[i]==0 && flag==1)	       //check if end of the current region is reached
    {
      meanR=meanR/currentLengthR;		       //compute the mean
      for(j=start;j<i;j++)			       //compute the variance of all elements of the last region
	varR+=(data[j][5]-meanR)*(data[j][5]-meanR);
      varR=varR/currentLengthR;
//printf("\n\n%d %d %f",start,i-1,varR);	       //uncomment to get the variance of each region and use that information to set the threshold in the next statement
      if(varR<0.3)				       //check if variance of last region<threshold or not. If <threshold, replace contents of indices corresponding to the region in indicesPang with all 0s
	for(j=start;j<i;j++)			       //This eliminates the smaller regions which is, in fact, just noise
	  indicesRang[j]=0;
      currentLengthR=0;				       //set all these to zeros so that they can be used again for the next region
      meanR=0;
      varP=0;
      flag=0;
    }
  }
*/
  //detecting the duration of change in yaw
  varThYang=0.02;				       //this is the variance threshold for change in yaw
  flag=0;					       //the flag is set when a new region is identified
  subflag=0;					       //the subflag is set when control is inside the new region
  interMean=0;					       //to find the mean of a window of samples inside the body of the current region
  abso;	 				 	       //used to find the absolute value as the function abs() does not work with floating point numbers
  for(i=0;i<(n-windowYang);i++)			       //this loop slides the window across the length of the entire dataset
  {
    sum=mean=variance=0;
    for(j=0;j<windowYang;j++)			       //this loop generates the window of samples for the current iteration
    {
      testArrayYang[j]=data[i+j][6];
      sum+=testArrayYang[j];
    }
    mean=sum/windowYang;			       //compute the mean of the window of samples
    for(j=0;j<windowYang;j++)
      variance+=(testArrayYang[j]-mean)*(testArrayYang[j]-mean);//you need the sum of the squares of the difference between each sample and the mean in order to calculate the variance
    variance=variance/windowYang;		       //compute the variance
    if(variance>varThYang && flag==0)		       //check if it is the beginning of a new region
    {
      flag=1;					       //set the flag as a new region has started
      subflag=1;				       //set the subflag as control must now go inside the region's body
    }
    if(subflag==1)
    {
      indicesYang[i]=i;
      interMean=0;
      for(j=0;j<windowYang;j++)
	interMean+=data[i+j][6];
      interMean=interMean/windowYang;
      if(interMean<0)				       //finding the absolute value of the intermediate mean - the strategy is to find the end point based on when exactly the mean of the chosen window goes
	abso=(-1)*interMean;			       //below a certain threshold. As the dataset has pitch variations that are both positive and negative, we must use the magnitude for thresholding.
      else
	abso=interMean;
      if(abso<0.001824)				       //use mean thresholding to identify if the end of the region has been reached. If the end is reached, reset the flags.
      {
	subflag=0;
	flag=0;
      }
    }
    //printf("%d %d %f\n",i,indicesYang[i],data[i][6]);
  }
/*
  //eliminate the noisy region from indicesYang
  flag=0;
  for(i=0;i<n;i++)
  { 
    if(indicesYang[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmYa++;
    }
    if(indicesYang[i]==0)			       //when indicesYang[i] becomes zero again (when current region ends), reset the flag
      flag=0;
//printf("%d %d %d\n",i,flag,nmYa);
  }
  int regionLengthYa[nmYa],currentLengthYa=0; 	       //regionLengthYa - store length of each region, currentLengthYa - length of the region that 'i' is currently in
  start=0;					       //start - starting point of current region
  for(i=0;i<nmYa;i++)
    regionLengthYa[i]=0;			       //initialize length of each region to 0
  flag=0;
  for(i=0;i<n;i++)
  {
    if(indicesYang[i]!=0 && flag==0)		       //check if beginning of a region - this condition is true only at the beginning of each region (first index of a region)
    {
      flag=1;					       //set flag as region entered
      start=i;
      meanYa+=data[i][6];
      currentLengthYa++;
    }
    else if(indicesYang[i]!=0 && flag==1)	       //check if 'i' is still in the region - this condition is true as long as 'i' is inside a region
    {
      meanYa+=data[i][6];
      currentLengthYa++;      
    }
    else if(indicesYang[i]==0 && flag==1)	       //check if end of the current region is reached
    {
      meanYa=meanYa/currentLengthYa;		       //compute the mean
      for(j=start;j<i;j++)			       //compute the variance of all elements of the last region
	varYa+=(data[j][6]-meanYa)*(data[j][6]-meanYa);
      varYa=varYa/currentLengthYa;
//printf("\n\n%d %d %f",start,i-1,varYa);	       //uncomment to get the variance of each region and use that information to set the threshold in the next statement
      if(varYa<0.06)				       //check if variance of last region<threshold or not. If <threshold, replace contents of indices corresponding to the region in indicesPang with all 0s
	for(j=start;j<i;j++)			       //This eliminates the smaller regions which is, in fact, just noise
	  indicesYang[j]=0;
      currentLengthYa=0;			       //set all these to zeros so that they can be used again for the next region
      meanYa=0;
      varYa=0;
      flag=0;
    }
  }
*/
  //calculate the time taken
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Windowing is now complete...\n\n");
  printf("Time at the end of windowing is %lds and %ldns.\n\n",(long int)tp2.tv_sec,(long int)tp2.tv_nsec);
  printf("Time taken to finish windowing and extract the indices is %lds and %ldns.\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)abs(tp2.tv_nsec-tp1.tv_nsec));

  //write all the indices that correspond to the object's motion to a text file
  fpt=fopen("indices.txt","w");
  if(fpt==NULL)
  {
    printf("Unable to open the file indices.txt for writing!\n");
    exit(0);
  }
  for(i=0;i<n;i++)
  {
    if(i==0)
      fprintf(fpt,"X\tY\tZ\tP\tR\tYa\n");
    fprintf(fpt,"%d\t%d\t%d\t%d\t%d\t%d\n",indicesXacc[i],indicesYacc[i],indicesZacc[i],indicesPang[i],indicesRang[i],indicesYang[i]);
  }
  fclose(fpt);
  printf("\nCheck the file 'indices.txt' to see the extracted indices.\n\n");
  printf("---------------------------------------------------------------------------------------------------------------------------------\n\n");

/////////////////////////////////////////////////MOTION TRACKING////////////////////////////////////////////////

  //start ticking just before motion tracking begins
  printf("Motion tracking begins now...\n\n");
  clock_gettime(CLOCK_REALTIME,&tp1);
  printf("Time at the beginning of tracking is %lds and %ldns.\n\n",(long int)tp1.tv_sec,(long int)tp1.tv_nsec);

  //identifying the starting and ending points of the regions of motion along the x-axis
  flag=0;
  nmX=0;
  for(i=0;i<n;i++)				       //we start off with identifying the starting and ending points of each region in the data (noisy regions have alredy been removed)
  { 
    if(indicesXacc[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmX++;
    }
    if(indicesXacc[i]==0)			       //when indicesXacc[i] becomes zero again (when current region ends), reset the flag
      flag=0;
  }
  printf("The number of regions in the filtered Xacc data is %d and their respective starting and ending points are: ",nmX);
  int startXacc[nmX],endXacc[nmX];		       //to store the starting and ending points of various regions in the data
  flag=0;
  nmX=0;
  for(i=0;i<n;i++)				       //this loop takes control through the data and updates startXacc and endXacc everytime a new region starts or an existing region ends
  {
    if(indicesXacc[i]!=0 && flag==0)		       //identifies the beginning of a region
    {
      flag=1;
      startXacc[nmX]=i;
    }
    if(indicesXacc[i]==0 && flag==1)		       //identifies the end of a region
    {
      flag=0;
      endXacc[nmX]=i;
      nmX++;
    }
  }
  for(i=0;i<nmX;i++)				       //prints the starting and ending points of the regions
    printf("(%d,%d) ",startXacc[i],endXacc[i]);
  printf("\n");

  //calculating the magnitude of motion along the x-axis
  float dispX[nmX];
  sum=0;
  for(i=0;i<nmX;i++)
  {
    velPrevious=0;
    dispX[i]=0;
    for(j=startXacc[i];j<=endXacc[i];j++)
    {
      velCurrent=data[j][1]*sampTime;
      velFinal=velPrevious+velCurrent;
      velAverage=(velFinal+velPrevious)/2;
      dispX[i]+=velAverage*sampTime;
//printf("%d\t%f\t%f\t%f\t%f\t%f\n",j,data[j][1],velCurrent,velFinal,velAverage,dispX[i]);
      velPrevious=velFinal;
      //velPrevious=velAverage;
    }
    printf("Displacement of region %d is %fm.\n",i+1,dispX[i]);
    sum+=dispX[i];
  }
  printf("The total displacement is %fm.\n\n",sum);

  //identifying the starting and ending points of the regions of motion along the y-axis
  flag=0;
  nmY=0;
  for(i=0;i<n;i++)				       //we start off with identifying the starting and ending points of each region in the data (noisy regions have alredy been removed)
  { 
    if(indicesYacc[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmY++;
    }
    if(indicesYacc[i]==0)			       //when indicesYacc[i] becomes zero again (when current region ends), reset the flag
      flag=0;
  }
  printf("The number of regions in the filtered Yacc data is %d and their respective starting and ending points are: ",nmY);
  int startYacc[nmY],endYacc[nmY];		       //to store the starting and ending points of various regions in the data
  flag=0;
  nmY=0;
  for(i=0;i<n;i++)				       //this loop takes control through the data and updates startYacc and endYacc everytime a new region starts or an existing region ends
  {
    if(indicesYacc[i]!=0 && flag==0)		       //identifies the beginning of a region
    {
      flag=1;
      startYacc[nmY]=i;
    }
    if(indicesYacc[i]==0 && flag==1)		       //identifies the end of a region
    {
      flag=0;
      endYacc[nmY]=i;
      nmY++;
    }
  }
  for(i=0;i<nmY;i++)				       //prints the starting and ending points of the regions
    printf("(%d,%d) ",startYacc[i],endYacc[i]);
  printf("\n");

  //calculating the magnitude of motion along the y-axis
  float dispY[nmY];
  sum=0;
  for(i=0;i<nmY;i++)
  {
    velPrevious=0;
    dispY[i]=0;
    for(j=startYacc[i];j<=endYacc[i];j++)
    {
      velCurrent=data[j][2]*sampTime;
      velFinal=velPrevious+velCurrent;
      velAverage=(velFinal+velPrevious)/2;
      dispY[i]+=velAverage*sampTime;
      velPrevious=velFinal;
      //velPrevious=velAverage;
//printf("%f\n",dispZ[i]);
    }
    printf("Displacement of region %d is %fm.\n",i+1,dispY[i]);
    sum+=dispY[i];
  }
  printf("The total displacement is %fm.\n\n",sum);

  //identifying the starting and ending points of the regions of motion along the z-axis
  flag=0;
  nmZ=0;
  for(i=0;i<n;i++)				       //we start off with identifying the starting and ending points of each region in the data (noisy regions have alredy been removed)
  { 
    if(indicesZacc[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmZ++;
    }
    if(indicesZacc[i]==0)			       //when indicesZacc[i] becomes zero again (when current region ends), reset the flag
      flag=0;
  }
  printf("The number of regions in the filtered Zacc data is %d and their respective starting and ending points are: ",nmZ);
  int startZacc[nmZ],endZacc[nmZ];		       //to store the starting and ending points of various regions in the data
  flag=0;
  nmZ=0;
  for(i=0;i<n;i++)				       //this loop takes control through the data and updates startZacc and endZacc everytime a new region starts or an existing region ends
  {
    if(indicesZacc[i]!=0 && flag==0)		       //identifies the beginning of a region
    {
      flag=1;
      startZacc[nmZ]=i;
    }
    if(indicesZacc[i]==0 && flag==1)		       //identifies the end of a region
    {
      flag=0;
      endZacc[nmZ]=i;
      nmZ++;
    }
  }
  for(i=0;i<nmZ;i++)				       //prints the starting and ending points of the regions
    printf("(%d,%d) ",startZacc[i],endZacc[i]);
  printf("\n");

  //calculating the magnitude of motion along the z-axis
  float dispZ[nmZ];
  sum=0;
  for(i=0;i<nmZ;i++)
  {
    velPrevious=0;
    dispZ[i]=0;
    for(j=startZacc[i];j<=endZacc[i];j++)
    {
      velCurrent=data[j][3]*sampTime;
      velFinal=velPrevious+velCurrent;
      velAverage=(velFinal+velPrevious)/2;
      dispZ[i]+=velAverage*sampTime;
      velPrevious=velFinal;
//printf("%f\n",dispZ[i]);
    }
    printf("Displacement of region %d is %fm.\n",i+1,dispZ[i]);
    sum+=dispZ[i];
  }
  printf("The total displacement is %fm.\n\n",sum);

  //identifying the starting and ending points of the regions in the pitch data
  flag=0;
  nmP=0;
  for(i=0;i<n;i++)				       //we start off with identifying the starting and ending points of each region in the data (noisy regions have alredy been removed)
  { 
    if(indicesPang[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmP++;
    }
    if(indicesPang[i]==0)			       //when indicesPang[i] becomes zero again (when current region ends), reset the flag
      flag=0;
  }
  printf("The number of regions in the filtered pitch data is %d and their respective starting and ending points are: ",nmP);
  int startPang[nmP],endPang[nmP];		       //to store the starting and ending points of various regions in the data
  flag=0;
  nmP=0;
  for(i=0;i<n;i++)				       //this loop takes control through the data and updates startPang and endPang everytime a new region starts or an existing region ends
  {
    if(indicesPang[i]!=0 && flag==0)		       //identifies the beginning of a region
    {
      flag=1;
      startPang[nmP]=i;
    }
    if(indicesPang[i]==0 && flag==1)		       //identifies the end of a region
    {
      flag=0;
      endPang[nmP]=i;
      nmP++;
    }
  }
  for(i=0;i<nmP;i++)				       //prints the starting and ending points of the regions
    printf("(%d,%d) ",startPang[i],endPang[i]);
  printf("\n");

  //calculating the magnitude of the displacement in pitch
  float dispP[nmP];
  sum=0;
  for(i=0;i<nmP;i++)
  {
    dispP[i]=0;
    for(j=startPang[i];j<=endPang[i];j++)
    {
      dispP[i]+=data[j][4]*sampTime;
//printf("%f\n",dispR[i]);
    }
    printf("Displacement of region %d is %frad.\n",i+1,dispP[i]);
    sum+=dispP[i];
  }
  printf("The total displacement is %frad.\n\n",sum);

  //identifying the starting and ending points of the regions in the roll data
  flag=0;
  nmR=0;
  for(i=0;i<n;i++)				       //we start off with identifying the starting and ending points of each region in the data (noisy regions have alredy been removed)
  { 
    if(indicesRang[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmR++;
    }
    if(indicesRang[i]==0)			       //when indicesRang[i] becomes zero again (when current region ends), reset the flag
      flag=0;
  }
  printf("The number of regions in the filtered roll data is %d and their respective starting and ending points are: ",nmR);
  int startRang[nmR],endRang[nmR];		       //to store the starting and ending points of various regions in the data
  flag=0;
  nmR=0;
  for(i=0;i<n;i++)				       //this loop takes control through the data and updates startRang and endRang everytime a new region starts or an existing region ends
  {
    if(indicesRang[i]!=0 && flag==0)		       //identifies the beginning of a region
    {
      flag=1;
      startRang[nmR]=i;
    }
    if(indicesRang[i]==0 && flag==1)		       //identifies the end of a region
    {
      flag=0;
      endRang[nmR]=i;
      nmR++;
    }
  }
  for(i=0;i<nmR;i++)				       //prints the starting and ending points of the regions
    printf("(%d,%d) ",startRang[i],endRang[i]);
  printf("\n");

  //calculating the magnitude of the displacement in roll
  float dispR[nmR];
  sum=0;
  for(i=0;i<nmR;i++)
  {
    dispR[i]=0;
    for(j=startRang[i];j<=endRang[i];j++)
    {
      dispR[i]+=data[j][5]*sampTime;
//printf("%f\n",dispR[i]);
    }
    printf("Displacement of region %d is %frad.\n",i+1,dispR[i]);
    sum+=dispR[i];
  }
  printf("The total displacement is %frad.\n\n",sum);

  //identifying the starting and ending points of the regions in the yaw data
  flag=0;
  nmYa=0;
  for(i=0;i<n;i++)				       //we start off with identifying the starting and ending points of each region in the data (noisy regions have alredy been removed)
  { 
    if(indicesYang[i]!=0 && flag==0)		       //the moment the variable 'i' enters a new region, #regions is incremented
    {
      flag=1;
      nmYa++;
    }
    if(indicesYang[i]==0)			       //when indicesYang[i] becomes zero again (when current region ends), reset the flag
      flag=0;
  }
  printf("The number of regions in the filtered yaw data is %d and their respective starting and ending points are: ",nmYa);
  int startYang[nmYa],endYang[nmYa];		       //to store the starting and ending points of various regions in the data
  flag=0;
  nmYa=0;
  for(i=0;i<n;i++)				       //this loop takes control through the data and updates startYang and endYang everytime a new region starts or an existing region ends
  {
    if(indicesYang[i]!=0 && flag==0)		       //identifies the beginning of a region
    {
      flag=1;
      startYang[nmYa]=i;
    }
    if(indicesYang[i]==0 && flag==1)		       //identifies the end of a region
    {
      flag=0;
      endYang[nmYa]=i;
      nmYa++;
    }
  }
  for(i=0;i<nmYa;i++)				       //prints the starting and ending points of the regions
    printf("(%d,%d) ",startYang[i],endYang[i]);
  printf("\n");

  //calculating the magnitude of the displacement in roll
  float dispYa[nmYa];
  sum=0;
  for(i=0;i<nmYa;i++)
  {
    dispYa[i]=0;
    for(j=startYang[i];j<=endYang[i];j++)
    {
      dispYa[i]+=data[j][6]*sampTime;
//printf("%f\n",dispYa[i]);
    }
    printf("Displacement of region %d is %frad.\n",i+1,dispYa[i]);
    sum+=dispYa[i];
  }
  printf("The total displacement is %frad.\n\n",sum);

  //calculate the time taken
  clock_gettime(CLOCK_REALTIME,&tp2);
  printf("Motion tracking is now complete...\n\n");
  printf("Time at the end of tracking is %lds and %ldns.\n\n",(long int)tp2.tv_sec,(long int)tp2.tv_nsec);
  printf("Time taken to finish tracking is %lds and %ldns.\n",(long int)(tp2.tv_sec-tp1.tv_sec),(long int)abs(tp2.tv_nsec-tp1.tv_nsec));

}
