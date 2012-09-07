#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

const int TPB=128;

// the kernel
__global__ void doublify(float * f_out,float * f, const int N){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<N){
    f_out[tid]=2.0*f[tid];
  }

}


// this is the Jacket SDK entry function.  It is analogous to a
// regular MATLAB MEX-function
err_t jktFunction(int nlhs,mxArray * plhs[],int nrhs,mxArray * prhs[]){

  if(nrhs!=1)
    return err("Usage: f_out = doublify(f)");

  //get a pointer to the input vector
  mxArray * m_f = prhs[0];

  //determine data type 
  mxClassID cls=jkt_class(m_f);

  //pointer to variable that will hold input vector size
  const mwSize * dims; 

  //get input vector size
  int status = jkt_dims(m_f,&dims);
  int N = dims[0];

  //create output vector
  //jkt_new(num rows, num cols, class, isComplex)
  mxArray * m_f_out = plhs[0]=jkt_new(N,1,cls,false);


  //get native pointer to input and output data
  float * f;
  float * f_out;

  //this looks like memory allocation, but really
  //all you're doing is extracting a pointer to the data
  //of the mxArray objects.
  jkt_mem((void**)&f,m_f);
  jkt_mem((void**)&f_out,m_f_out);

  // set the thread launch configuration
  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((N+TPB-1)/TPB,1,1);

  //call the kernel
  doublify<<<GRIDS,BLOCKS>>>(f_out,f,N);


  return errNone;
}
