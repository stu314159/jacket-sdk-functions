#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

__global__ void atomicAddUpdateKernel(float * A,float * A_update,
				      float * index_vec,float * update_vec,
				      int num,int num_idx){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if(tid<num){
    A_update[tid] = A[tid];
    if(tid<num_idx){
      int indx = (int)index_vec[tid] - 1;//make ones-based
      atomicAdd(&A_update[indx],update_vec[tid]);
    }
  }

}

err_t jktFunction(int nlhs, mxArray *plhs[],int nrhs, mxArray *prhs[]){

  if(nrhs != 3)
    return err("Usage: A_upd = vecUpdateAtomicAdd(A,ind_vec,update_vec)");

  mxArray * m_A = prhs[0];
  mxArray * m_ind_vec = prhs[1];
  mxArray * m_update_vec = prhs[2];

  mxClassID cls = jkt_class(m_A);

  const mwSize *dims;
  int in_dim = jkt_dims(m_A,&dims);

  int A_rows = dims[0];
  int A_cols = dims[1];

  in_dim = jkt_dims(m_ind_vec,&dims);

  int upd_rows = dims[0];
  int upd_cols = dims[1];

  int num_upd;
  if(upd_rows >= upd_cols){
    num_upd = upd_rows;
  }else{
    num_upd = upd_cols;
  }

  mxArray * A_upd = plhs[0] = jkt_new(A_rows,A_cols,cls,false);

  float * d_A, * d_A_upd, * d_ind_vec, * d_update_vec;

  TRY(jkt_mem((void**)&d_A,m_A));
  TRY(jkt_mem((void**)&d_A_upd,A_upd));
  TRY(jkt_mem((void**)&d_ind_vec,m_ind_vec));
  TRY(jkt_mem((void**)&d_update_vec,m_update_vec));

  unsigned num= A_rows*A_cols;
  unsigned threads = 256;
  unsigned blocks = (num+256)/256;

  //call kernel here...
  atomicAddUpdateKernel<<<blocks,threads>>>(d_A,d_A_upd,d_ind_vec,d_update_vec,
					    num,num_upd);

  return errNone;


}
