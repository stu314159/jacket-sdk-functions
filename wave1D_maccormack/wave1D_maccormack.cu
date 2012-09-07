#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

const int TPB=128;



// no shared memory
__global__ void wave1Dmac_step1(double * f_tmp1, double * f_in,
				double u, double dt, double dx,
				int N){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<N){
    int x_p = tid+1;
    if(x_p == N) x_p = 0;
    
    double f_tmp = f_in[tid];
    f_tmp1[tid]= f_tmp - u*(dt/dx)*(f_in[x_p] - f_tmp);

  }
}

__global__ void wave1Dmac_step2(double * f_next, double * f_tmp1,
				double * f_in, double u, double dt,
				double dx, int N){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<N){
    int x_m = tid-1;
    if(x_m <0) x_m = N-1;

    double ft1_tmp = f_tmp1[tid];
    f_next[tid]=0.5*(f_in[tid]+ft1_tmp - u*(dt/dx)*(ft1_tmp-f_tmp1[x_m]));

  }
}

err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){


  if(nrhs!=4)
    return err("Usage: f_next = wave1D_maccormack(f_in,u,dt,dx)");

  mxArray * m_f_in = prhs[0];
  double u = mxGetScalar(prhs[1]);
  double dt = mxGetScalar(prhs[2]);
  double dx = mxGetScalar(prhs[3]);

  mxClassID cls = jkt_class(m_f_in);
  const mwSize * dims;

  int status = jkt_dims(m_f_in,&dims);
  int M = dims[0];
  int N = dims[1];

  

  mxArray * m_f_next = plhs[0]=jkt_new(M,N,cls,false);

  mxArray * m_ftmp1 = jkt_new(M,N,cls,false);


  double * f_in;
  double * f_next;
  double * f_tmp1;

  jkt_mem((void**)&f_in, m_f_in);
  jkt_mem((void**)&f_next,m_f_next);
  jkt_mem((void**)&f_tmp1,m_ftmp1);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((N+TPB-1)/TPB,1,1);

  wave1Dmac_step1<<<GRIDS,BLOCKS>>>(f_tmp1,f_in,u,dt,dx,M*N);
  wave1Dmac_step2<<<GRIDS,BLOCKS>>>(f_next,f_tmp1,f_in,u,dt,dx,M*N);

  return errNone;

}
