#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define TPB 128

__global__ void velBC(float * fIn_d, int * bc_nl, float * rho_d, float * ux_d,
		      float * uy_d, float * uz_d,float * ux_p,
		      float * uy_p, float * uz_p,const float * ex,
		      const float * ey, const float * ez, const float * w,
		      const int numBC,const int nnodes, const int numSpd){
  int l_tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(l_tid<numBC){
    int tid = bc_nl[l_tid]-1;
    float rho = rho_d[tid];
    float dx = ux_p[l_tid]-ux_d[tid];
    float dy = uy_p[l_tid]-uy_d[tid];
    float dz = uz_p[l_tid]-uz_d[tid];
    float cu;
    for(int spd=1;spd<numSpd;spd++){
      cu = 3.0*(ex[spd]*dx+ey[spd]*dy+ez[spd]*dz);
      fIn_d[tid+nnodes*numSpd]+=w[spd]*rho*cu;
    }
    ux_d[tid]=ux_p[l_tid]; uy_d[tid]=uy_p[l_tid]; uz_d[tid]=uz_p[l_tid];

  }
}


err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]){


  if(nrhs!=14)
    return err("Usage: velocityBC3D(fIn,bc_nl,rho,ux,uy,uz,ux_p,uy_p,uz_p,ex,ey,ez,w,numBC)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_bc_nl = prhs[1];
  mxArray * m_rho = prhs[2];
  mxArray * m_ux = prhs[3];
  mxArray * m_uy = prhs[4];
  mxArray * m_uz = prhs[5];
  mxArray * m_ux_p = prhs[6];
  mxArray * m_uy_p = prhs[7];
  mxArray * m_uz_p = prhs[8];
  mxArray * m_ex = prhs[9];
  mxArray * m_ey = prhs[10];
  mxArray * m_ez = prhs[11];
  mxArray * m_w = prhs[12];
  int numBC = mxGetScalar(prhs[13]);

  const mwSize * dims;
  int stat = jkt_dims(m_fIn,&dims);

  int nnodes = dims[0];
  int numSpd = dims[1];

  //get pointers to device data
  float * fIn; jkt_mem((void**)&fIn,m_fIn);
  int * bc_nl; jkt_mem((void**)&bc_nl,m_bc_nl);
  float * rho_d; jkt_mem((void**)&rho_d,m_rho);
  float * ux_d; jkt_mem((void**)&ux_d,m_ux);
  float * uy_d; jkt_mem((void**)&uy_d,m_uy);
  float * uz_d; jkt_mem((void**)&uz_d,m_uz);
  float * ux_p; jkt_mem((void**)&ux_p,m_ux_p);
  float * uy_p; jkt_mem((void**)&uy_p,m_uy_p);
  float * uz_p; jkt_mem((void**)&uz_p,m_uz_p);
  float * ex; jkt_mem((void**)&ex,m_ex);
  float * ey; jkt_mem((void**)&ey,m_ey);
  float * ez; jkt_mem((void**)&ez,m_ez);
  float * w; jkt_mem((void**)&w,m_w);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  velBC<<<GRIDS,BLOCKS>>>(fIn,bc_nl,rho_d,ux_d,uy_d,uz_d,
			  ux_p,uy_p,uz_p,ex,ey,ez,w,numBC,nnodes,numSpd);

  return errNone;
}
