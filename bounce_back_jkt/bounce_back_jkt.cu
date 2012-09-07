#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define TPB 128

__global__ void bounce_back(float * fOut, float * fIn,
			    int * snl, 
			    int * bb_spd, int nnodes){

  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int spd=blockIdx.y;
  if(tid<nnodes){
    if(snl[tid]==1){
      int spd_src = bb_spd[spd] - 1;//correct for ones-based array value
      fOut[spd*nnodes+tid]=fIn[spd_src*nnodes+tid];
    }
  }


}



err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){


  if(nrhs!=4)
    return err("Usage: bounce_back_jkt(fOut,fIn,snl,bb_spd)");

  mxArray * m_fOut = prhs[0];
  mxArray * m_fIn = prhs[1];
  mxArray * m_snl = prhs[2];
  mxArray * m_bbspd = prhs[3];

  mxClassID cls = jkt_class(m_fIn);

  const mwSize * dims;
  int stat = jkt_dims(m_fIn,&dims);

  int nnodes = dims[0];
  int numSpd = dims[1];

  //create the output array.
  //mxArray * m_fOut_new = plhs[0] = jkt_new(nnodes,numSpd,cls,false);

  //get pointers to all the device data...
  //float * fOut_new_d;
  float * fOut_d;
  float * fIn_d;
  int * snl_d;
  int * bbspd_d;

  //direct these pointers to device data
  //jkt_mem((void**)&fOut_new_d,m_fOut_new);
  jkt_mem((void**)&fOut_d,m_fOut);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&snl_d,m_snl);
  jkt_mem((void**)&bbspd_d,m_bbspd);

  //define thread execution configuration
  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,numSpd,1);
  bounce_back<<<GRIDS,BLOCKS>>>(fOut_d,
				fIn_d,snl_d,bbspd_d,nnodes);


  return errNone;

}
