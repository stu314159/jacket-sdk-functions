#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define TPB 128

__global__ void stream(float * fIn, float * fOut, int * stm, int nnodes){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int spd = blockIdx.y;
  if(tid<nnodes){
    int stream_tgt = stm[spd*nnodes+tid] - 1;
    fIn[spd*nnodes+stream_tgt]=fOut[spd*nnodes+tid];

  }
} 

err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs,mxArray * prhs[]){

  if(nrhs!=2)
    return err("Usage: fIn = stream_jkt(fOut,stm)");

  mxArray * m_fOut = prhs[0];
  mxArray * m_stm = prhs[1];


  mxClassID cls = jkt_class(m_fOut);

  const mwSize * dims;
  int stat = jkt_dims(m_fOut,&dims);

  int nnodes = dims[0];
  int numSpd = dims[1];

  //create output array
  mxArray * m_fIn_new = plhs[0] = jkt_new(nnodes,numSpd,cls,false);

  //get pointers to all device data
  float * fIn_d;
  float * fOut_d;
  int * stm_d;

  jkt_mem((void**)&fIn_d,m_fIn_new);
  jkt_mem((void**)&fOut_d,m_fOut);
  jkt_mem((void**)&stm_d,m_stm);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,numSpd,1);

  stream<<<GRIDS,BLOCKS>>>(fIn_d,fOut_d,stm_d,nnodes);



  return errNone;
}
