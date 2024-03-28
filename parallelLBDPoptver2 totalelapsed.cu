#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<stdlib.h>
#include "dcmtk/dcmimgle/dcmimage.h"
#include<algorithm>
#include<stdio.h>


#define TILE_WIDTH 16
#define TILE_HEIGHT 16

typedef  unsigned short int USint;

using namespace std;

__constant__ USint multiple2[16];

__constant__ char x[8];

__constant__ char y[8];


__global__ void getLBDP(USint *pdata,int h, int w, unsigned char *out) {

	int tx=blockIdx.x*blockDim.x+threadIdx.x;
	int ty=blockIdx.y*blockDim.y+threadIdx.y;

	

	__shared__ USint bpdata[TILE_HEIGHT+2][TILE_WIDTH+2];

	bpdata[threadIdx.y+1][threadIdx.x+1]=pdata[ty*w+tx];

	USint bitp[16];



	if(threadIdx.x==0)
	{
		if(tx != 0)
		{
			bpdata[threadIdx.y+1][0]=pdata[ty*w+tx-1];
			if(threadIdx.y==0 && ty!=0)
				bpdata[0][0]=pdata[(ty-1)*w+tx-1];
		}
		
	}
	if(threadIdx.x==TILE_WIDTH-1)
	{
		if(tx != w-1) {
			bpdata[threadIdx.y+1][threadIdx.x+2]=pdata[ty*w+tx+1];
			if(threadIdx.y==TILE_HEIGHT-1 && ty!=h-1)
				bpdata[TILE_HEIGHT+1][TILE_WIDTH+1]=pdata[(ty+1)*w+tx+1];
		}
	}

	if(threadIdx.y==0)
	{
		if(ty != 0)
		{

			bpdata[0][threadIdx.x+1]=pdata[(ty-1)*w+tx];
			if(threadIdx.x == TILE_WIDTH-1 && tx!=w-1)
				bpdata[0][TILE_WIDTH+1]=pdata[(ty-1)*w+tx+1];
		}
	}
	if(threadIdx.y==TILE_HEIGHT-1)
	{
		if(ty!=h-1)
		{
			bpdata[threadIdx.y+2][threadIdx.x+1]=pdata[(ty+1)*w+tx];
			if(threadIdx.x==0 && tx!=0)
				bpdata[TILE_HEIGHT+1][0]=pdata[(ty+1)*w+tx-1];
		}
	} 


			


	if(ty==0 || ty==h-1 || tx==0 || tx==w-1)
		return;

	__syncthreads();
	
			
	USint centre;
	centre=bpdata[threadIdx.y+1][threadIdx.x+1];

	USint cur;
	USint pow;
	USint btot;
	btot=0;


	for(int l=0;l<16;l++)
		bitp[l]=0;
        for(int l=0;l<16;l++)
	{
		pow=1;

		for(int k=0;k<8;k++)
		{
			cur=bpdata[threadIdx.y+1+y[k]][threadIdx.x+1+x[k]];
			if(cur & multiple2[l])
				bitp[l] = bitp[l] + pow;

			pow = pow*2;
		}
	}
	for(int l=15;l>=0;l--)
	{	
		if(bitp[l] > centre)
			btot = btot+multiple2[15-l];
	}

	out[(ty-1)*(w-2)+(tx-1)]=(unsigned char)(btot>>8);

}

					


int main(int argc, char *argv[]) {
	
	cudaEvent_t pstart,pstop;

	float totalelapsedTime;
	for (int i = 0; i < 50; ++i) {
	cudaEventCreate(&pstart);
	cudaEventCreate(&pstop);

	cudaEventRecord(pstart,0);

	DicomImage *img=new DicomImage("../images/512x512CT.dcm");

	if(img != NULL && img->getStatus()==EIS_Normal) {
		if(img->isMonochrome()) {
			img->setMinMaxWindow();
			
			int h=img->getHeight();
			int w=img->getWidth();


			USint *pdata = (USint *)new USint[h*w];

			img->getOutputData(pdata,w*h*sizeof(USint));


			USint *d_pdata;
			cudaMalloc((void **)&d_pdata,w*h*sizeof(USint));

			cudaMemcpy(d_pdata,pdata,w*h*sizeof(USint),cudaMemcpyHostToDevice);



                     	unsigned char *out;

			out=(unsigned char *)calloc((h-2)*(w-2),sizeof(unsigned char));
			
			unsigned char *d_out;
			cudaMalloc((void **)&d_out,((h-2)*(w-2)*sizeof(unsigned char)));

			USint *h_multiple2;
			h_multiple2=(USint *)calloc(sizeof(USint)*8,sizeof(USint));

			h_multiple2[0]=1;
			for(int l=1;l<16;l++) 
			{
				h_multiple2[l]=h_multiple2[l-1]<<1;
				//printf("%d   \n",multiple2[l]);		
			}
			char h_x[8]={1,1,0,-1,-1,-1,0,1};
			char h_y[8]={0,-1,-1,-1,0,1,1,1};

			cudaMemcpyToSymbol(multiple2,h_multiple2,sizeof(USint)*8*sizeof(USint),0,cudaMemcpyHostToDevice);

			cudaMemcpyToSymbol(x,h_x,sizeof(char)*8,0,cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(y,h_y,sizeof(char)*8,0,cudaMemcpyHostToDevice);


			cudaEvent_t kstart, kstop;

			dim3 blk(std::max(w/TILE_WIDTH,1),std::max(h/TILE_HEIGHT,1));
			dim3 thrd(std::min(w,TILE_WIDTH),std::min(h,TILE_HEIGHT));
			//dim3 thrd(3,3);

			cudaEventCreate(&kstart);
			cudaEventCreate(&kstop); 

			cudaEventRecord(kstart, 0);
			getLBDP<<<blk,thrd>>>(d_pdata,h,w,d_out);
			
			cudaEventRecord(kstop, 0);
			cudaEventSynchronize(kstop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime, kstart, kstop);			
			
			printf("Kernel Elapsed time =%f\n",elapsedTime);

			cudaMemcpy(out,d_out,((h-2)*(w-2)*sizeof(unsigned char)),cudaMemcpyDeviceToHost);


			delete out;
		}
	}

	cudaEventRecord(pstop, 0);
	cudaEventSynchronize(pstop);
	cudaEventElapsedTime(&totalelapsedTime, pstart, pstop);
	printf("Program Elapsed time =%f\n",totalelapsedTime);
}
	return 0;
}
