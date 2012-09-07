#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 128

__global__ void D3Q19_getRho(const float * fIn, float * rho, const int nnodes){

int tid=threadIdx.x+blockIdx.x*blockDim.x;

 if(tid <nnodes){
   float rho_t = 0.;
   for(int spd=0;spd<19;spd++){
     rho_t+=fIn[spd*nnodes+tid];
   }
   rho[tid]=rho_t;
 }
}

err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]){

  if(nrhs!=3)
    return err("Usage: D3Q19_getRho(fIn, rho, nnodes)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_rho = prhs[1];
  int nnodes = mxGetScalar(prhs[2]);


  float * fIn;
  float * rho;

  jkt_mem((void**)&fIn, m_fIn);
  jkt_mem((void**)&rho, m_rho);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  D3Q19_getRho<<<GRIDS,BLOCKS>>>(fIn,rho,nnodes);


  return errNone;
}
