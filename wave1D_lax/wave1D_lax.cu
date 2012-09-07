#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

const int TPB=128;


// simple kernel
__global__ void wave1Dlax(double * f_next, double * f, double u, 
			  double dt, double dx, int N){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<N){
    //get DOF number for neighbor nodes
    //and apply periodic boundary conditions
    int x_p = tid+1;
    if(x_p ==N)
      x_p = 0;
    int x_m = tid-1;
    if(x_m<0)
      x_m = N-1;

    double f_p = f[x_p];
    double f_m = f[x_m];

    //Lax-method 
    f_next[tid]=0.5*(f_p + f_m) - (u*dt/(2.*dx))*(f_p - f_m);

  }
}

err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){


  if(nrhs!=4)
    return err("Usage: f_next = wave1D_lax(f_in,u,dt,dx)");

  mxArray * m_f_in = prhs[0];
  double u = mxGetScalar(prhs[1]);
  double dt = mxGetScalar(prhs[2]);
  double dx = mxGetScalar(prhs[3]);

  mxClassID cls = jkt_class(m_f_in);
  const mwSize * dims;

  int status = jkt_dims(m_f_in,&dims);
  int M = dims[0];
  int N = dims[1];

  //create return value object and load pointer into LHS array
  mxArray * m_f_next = plhs[0]=jkt_new(M,N,cls,false);

  //declare device pointer variables
  double * f_in;
  double * f_next;

  //direct pointers to device data
  jkt_mem((void**)&f_in, m_f_in);
  jkt_mem((void**)&f_next,m_f_next);

  //configure thread execution configuration
  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((N+TPB-1)/TPB,1,1);

  //launch kernel
  wave1Dlax<<<GRIDS,BLOCKS>>>(f_next,f_in,u,dt,dx,M*N);

  return errNone;

}
