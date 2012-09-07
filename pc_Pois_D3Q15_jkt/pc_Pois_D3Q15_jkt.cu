#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define SPDS 15
#define TPB 64

__global__ void pre_collideD3Q15(float * fEq,float * fIn,
				 float * ex, float * ey,
				 float * ez, 
				 float * ux_ip,float * ux_op,
				 float * w, int * inl,
				 int * onl, int nnodes){
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
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
    if(inl[tid]==1){
      float dx = ux_ip[tid]-ux;
      float dy = -uy;
      float dz =-uz;

      for(int spd=0;spd<SPDS;spd++){
	cu= 3.0*(ex[spd]*dx+ey[spd]*dy+ez[spd]*dz);
	fIn_s[threadIdx.x][spd]+=w[spd]*rho*cu;
	//write updated fIn back to global memory.
	fIn[spd*nnodes+tid]=fIn_s[threadIdx.x][spd];
      }
    
      ux +=dx;
      uy +=dy;
      uz +=dz;
    }

    if(onl[tid]==1){
      float dx = ux_op[tid]-ux;
      float dy = -uy;
      float dz =-uz;

      for(int spd=0;spd<SPDS;spd++){
	cu= 3.0*(ex[spd]*dx+ey[spd]*dy+ez[spd]*dz);
	fIn_s[threadIdx.x][spd]+=w[spd]*rho*cu;
	//write updated fIn back to global memory.
	fIn[spd*nnodes+tid]=fIn_s[threadIdx.x][spd];
      }
    
      ux +=dx;
      uy +=dy;
      uz +=dz;
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


err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]){

  if(nrhs!=10)
    return err("Usage: pc_Pois_D3Q15_jkt(fIn,fEq,ex,ey,ez,ux_ip,ux_op,w,inl,onl");

  mxArray * m_fIn = prhs[0];
  mxArray * m_fEq = prhs[1];
  mxArray * m_ex = prhs[2];
  mxArray * m_ey = prhs[3];
  mxArray * m_ez = prhs[4];
  mxArray * m_ux_ip = prhs[5];
  mxArray * m_ux_op = prhs[6];
  mxArray * m_w = prhs[7];
  mxArray * m_inl = prhs[8];
  mxArray * m_onl = prhs[9];

  mxClassID cls=jkt_class(m_fIn);

  const mwSize * dims;
  int stat = jkt_dims(m_fIn,&dims);

  int nnodes = dims[0];

  float * fEq_d;
  float * fIn_d;
  float * ex_d;
  float * ey_d;
  float * ez_d;
  float * ux_ip_d;
  float * ux_op_d;
  float * w_d;
  int * inl_d;
  int * onl_d;

  jkt_mem((void**)&fEq_d,m_fEq);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&ex_d,m_ex);
  jkt_mem((void**)&ey_d,m_ey);
  jkt_mem((void**)&ez_d,m_ez);
  jkt_mem((void**)&w_d,m_w);
  jkt_mem((void**)&ux_ip_d,m_ux_ip);
  jkt_mem((void**)&ux_op_d,m_ux_op);
  jkt_mem((void**)&inl_d,m_inl);
  jkt_mem((void**)&onl_d,m_onl);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  pre_collideD3Q15<<<GRIDS,BLOCKS>>>(fEq_d,fIn_d,ex_d,ey_d,
				     ez_d,ux_ip_d,ux_op_d,
				     w_d,inl_d,onl_d,nnodes);

  return errNone;

}

