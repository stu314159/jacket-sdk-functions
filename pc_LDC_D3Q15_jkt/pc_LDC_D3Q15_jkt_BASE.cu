#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define SPDS 15
#define TPB 64

__global__ void pre_collideD3Q15(float * fIn_new, float * fEq,float * fIn,
				 float * ex, 
				 float * ey, float * ez, float * ux_p,
				 float * uy_p, float * uz_p, float * w,
				 int * bc_nl, int nnodes){

  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    //load fIn data into shared memory for convenient access
    __shared__ float fIn_s[TPB][SPDS];
    for(int spd=0;spd<SPDS;spd++){
      fIn_s[threadIdx.x][spd]=fIn[spd*nnodes+tid];
    }
    //get macroscopic velocity and density.
    float ux = 0.; float uy = 0.; float uz = 0.; float rho = 0.;
    float f_tmp; float cu;
    for(int spd=0;spd<SPDS; spd++){
      f_tmp = fIn_s[threadIdx.x][spd];
      rho+=f_tmp;
      ux+=ex[spd]*f_tmp;
      uy+=ey[spd]*f_tmp;
      uz+=ez[spd]*f_tmp;
    }
    //yes, I know, I should ensure rho not equal zero...
    ux = ux/rho;
    uy = uy/rho;
    uz = uz/rho;

    //if I'm a boundary node, apply bc.
    if(bc_nl[tid]==1){
      float dx = ux_p[tid]-ux;
      float dy = uy_p[tid]-uy;
      float dz = uz_p[tid]-uz;

      for(int spd=0;spd<SPDS;spd++){
	cu= 3.0*(ex[spd]*dx+ey[spd]*dy+ez[spd]*dz);
	fIn_s[threadIdx.x][spd]+=w[spd]*rho*cu;
	//write updated fIn back to global memory.
	fIn_new[spd*nnodes+tid]=fIn_s[threadIdx.x][spd];
      }
    
      ux +=dx;
      uy +=dy;
      uz +=dz;
    }else{
      //if not a lid node, just copy back to fIn_new
      for(int spd=0;spd<SPDS;spd++){
	fIn_new[spd*nnodes+tid]=fIn_s[threadIdx.x][spd];
      }
    }

    //now, compute fEq
    for(int spd=0;spd<SPDS;spd++){
      cu = 3.0*(ex[spd]*ux+ey[spd]*uy+ez[spd]*uz);
      fEq[nnodes*spd+tid]=w[spd]*rho*(1.+cu+0.5*(cu*cu)-
				      1.5*(ux*ux+
					   uy*uy+
					   uz*uz));
    }    

    

  }
}


err_t jktFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray * prhs[]){


  if(nrhs != 9)
    return err("Usage: [fIn,fEq]=pc_D3Q15_jkt(fIn,ex,ey,ez,ux_p,uy_p,uz_p,w,bc_nl)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_ex = prhs[1];
  mxArray * m_ey = prhs[2];
  mxArray * m_ez = prhs[3];
  mxArray * m_uxp = prhs[4];
  mxArray * m_uyp = prhs[5];
  mxArray * m_uzp = prhs[6];
  mxArray * m_w = prhs[7];
  mxArray * m_bcnl = prhs[8];

  mxClassID cls = jkt_class(m_fIn);

  const mwSize * dims;
  int stat = jkt_dims(m_fIn,&dims);

  int nnodes = dims[0]; //<-- do some error checking on this?
  int numSpd = dims[1];

  mxArray * m_fIn_new = plhs[0] = jkt_new(nnodes,numSpd,cls,false);
  mxArray * m_fEq = plhs[1] = jkt_new(nnodes,numSpd,cls,false); 
  // mxArray * m_fIn_new = plhs[0]= jkt_new(numSpd,nnodes,cls,false);
  // mxArray * m_fEq = plhs[1] = jkt_new(numSpd,nnodes,cls,false);

  //get pointers to all of the kernel arguments
  
  float * fIn_new_d;
  float * fEq_d;
  float * fIn_d;
  float * ex_d;
  float * ey_d;
  float * ez_d;
  float * w_d;
  float * uxp_d;
  float * uyp_d;
  float * uzp_d;
  int * bcnl_d;

  jkt_mem((void**)&fIn_new_d,m_fIn_new);
  jkt_mem((void**)&fEq_d,m_fEq);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&ex_d,m_ex);
  jkt_mem((void**)&ey_d,m_ey);
  jkt_mem((void**)&ez_d,m_ez);
  jkt_mem((void**)&w_d,m_w);
  jkt_mem((void**)&uxp_d,m_uxp);
  jkt_mem((void**)&uyp_d,m_uyp);
  jkt_mem((void**)&uzp_d,m_uzp);
  jkt_mem((void**)&bcnl_d,m_bcnl);

  //define thread execution configuration

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  pre_collideD3Q15<<<GRIDS,BLOCKS>>>(fIn_new_d,fEq_d,fIn_d,ex_d,ey_d,ez_d,
		   uxp_d,uyp_d,uzp_d,w_d,bcnl_d,nnodes);



  return errNone;

}
