#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define TPB 128


__global__ void comp_fEq(float * fEq,float * rho_d, float * ux_d, 
			 float * uy_d, float * uz_d,
			 const float * ex_d, const float * ey_d,
			 const float * ez_d, const float * w_d,
			 const int nnodes){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int spd = blockIdx.y;
  if(tid<nnodes){
    float ux,uy,uz,ex,ey,ez,w,cu;
    
    ux = ux_d[tid]; uy=uy_d[tid];uz = uz_d[tid]; 
    ex = ex_d[spd]; ey = ey_d[spd]; ez = ez_d[spd];
    w = w_d[spd];
    cu = 3.0*(ex*ux+ey*uy+ez*uz);
    fEq[tid+spd*nnodes]=w*rho_d[tid]*(1.0+cu+(0.5)*cu*cu - 
				    1.5*(ux*ux+uy*uy+uz*uz));


  }
}




err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]){

  if(nrhs!=9)
    return err("Usage: fEq3D(fEq,rho,ux,uy,uz,ex,ey,ez,w)");

  mxArray * m_fEq = prhs[0];
  mxArray * m_rho = prhs[1];
  mxArray * m_ux = prhs[2];
  mxArray * m_uy = prhs[3];
  mxArray * m_uz = prhs[4];
  mxArray * m_ex = prhs[5];
  mxArray * m_ey = prhs[6];
  mxArray * m_ez = prhs[7];
  mxArray * m_w = prhs[8];

  const mwSize * dims;
  int stat = jkt_dims(m_fEq,&dims);

  int nnodes = dims[0];
  int numSpd = dims[1];

  //get pointers to device data
  float * fEq; jkt_mem((void**)&fEq,m_fEq);
  float * rho; jkt_mem((void**)&rho,m_rho);
  float * ux; jkt_mem((void**)&ux,m_ux);
  float * uy; jkt_mem((void**)&uy,m_uy);
  float * uz; jkt_mem((void**)&uz,m_uz);
  float * ex; jkt_mem((void**)&ex,m_ex);
  float * ey; jkt_mem((void**)&ey,m_ey);
  float * ez; jkt_mem((void**)&ez,m_ez);
  float * w; jkt_mem((void**)&w,m_w);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,numSpd,1);
  comp_fEq<<<GRIDS,BLOCKS>>>(fEq,rho,ux,uy,uz,ex,ey,ez,w,nnodes);

  return errNone;
}
