#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

const int TPB=128;

__global__ void wave1D_rusanov1(double * f_nm, double * f_in, 
				double nu,int N){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<N){
    int x_p = tid+1;
    if(x_p==N) x_p=0;

    double fp = f_in[x_p];
    double f = f_in[tid];
    f_nm[tid]=0.5*(fp+f)-(nu/3.)*(fp-f);

  }
}


__global__ void wave1D_rusanov2(double * f_tmp,double * f_nm, 
				double * f_in, double nu, int N){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<N){
    int x_m = tid-1;
    if(x_m<0) x_m = (N-1);
    f_tmp[tid]=f_in[tid]-(2.*nu/3.)*(f_nm[tid]-f_nm[x_m]);

  }
}

__global__ void wave1D_rusanov3(double * f_next,double * f_tmp, 
				double * f_in, double nu,
				double omega, int N){
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<N){
    int x_2m=tid-2;
    if(x_2m<0) x_2m+=N;
    int x_m = tid-1;
    if(x_m<0) x_m+=N;

    int x_p = tid+1;
    if(x_p>(N-1)) x_p-=N;

    int x_2p = tid+2;
    if(x_2p>(N-1)) x_2p-=N;

    double f_2m = f_in[x_2m];
    double f_m = f_in[x_m];
    double f = f_in[tid];
    double f_p = f_in[x_p];
    double f_2p = f_in[x_2p];

    f_next[tid]=f-(nu/24.)*(-2.*f_2p+7.*f_p - 7.*f_m+2.*f_2m)
      -(3.*nu/8.)*(f_tmp[x_p]-f_tmp[x_m])
      -(omega/24.)*(f_2p - 4.*f_p + 6.*f - 4.*f_m + f_2m);


  }
}

err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){


  if(nrhs!=3)
    return err("Usage: f_next = wave1D_rusanov(f_in,nu,omega)");

  mxArray * m_f_in = prhs[0];
  double nu = mxGetScalar(prhs[1]);
  double omega = mxGetScalar(prhs[2]);
  //double dx = mxGetScalar(prhs[3]);

  mxClassID cls = jkt_class(m_f_in);
  const mwSize * dims;

  int status = jkt_dims(m_f_in,&dims);
  int M = dims[0];
  int N = dims[1];

  

  mxArray * m_f_next = plhs[0]=jkt_new(M,N,cls,false);

  mxArray * m_ftmp1 = jkt_new(M,N,cls,false);
  mxArray * m_fnm = jkt_new(M,N,cls,false);


  double * f_in;
  double * f_next;
  double * f_tmp1;
  double * f_nm;

  jkt_mem((void**)&f_in, m_f_in);
  jkt_mem((void**)&f_next,m_f_next);
  jkt_mem((void**)&f_tmp1,m_ftmp1);
  jkt_mem((void**)&f_nm,m_fnm);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((N+TPB-1)/TPB,1,1);

  wave1D_rusanov1<<<GRIDS,BLOCKS>>>(f_nm,f_in,nu,N*M);
  wave1D_rusanov2<<<GRIDS,BLOCKS>>>(f_tmp1,f_nm,f_in,nu,N*M);
  wave1D_rusanov3<<<GRIDS,BLOCKS>>>(f_next,f_tmp1,f_in,nu,omega,N*M);

  return errNone;

}
