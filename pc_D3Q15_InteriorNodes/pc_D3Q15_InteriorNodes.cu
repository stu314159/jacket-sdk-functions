#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 128

__global__ void pc_D3Q15_IN(float * fIn, float * fEq, float * rho_d,
			    float * ux_d, float * uy_d, float * uz_d,
			    int * INL, const int numIN, const int nnodes){
  int tid_nd = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid_nd<numIN){
    int tid = INL[tid_nd]-1;

    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
    float cu;
    //load the data into registers
    f0=fIn[tid]; f1=fIn[nnodes+tid];
    f2=fIn[2*nnodes+tid]; f3=fIn[3*nnodes+tid];
    f4=fIn[4*nnodes+tid]; f5=fIn[5*nnodes+tid];
    f6=fIn[6*nnodes+tid]; f7=fIn[7*nnodes+tid];
    f8=fIn[8*nnodes+tid]; f9=fIn[9*nnodes+tid];
    f10=fIn[10*nnodes+tid]; f11=fIn[11*nnodes+tid];
    f12=fIn[12*nnodes+tid]; f13=fIn[13*nnodes+tid];
    f14=fIn[14*nnodes+tid];
    //compute density and velocity
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy/=rho;
    float uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz/=rho;

    ux_d[tid]=ux;
    uy_d[tid]=uy;
    uz_d[tid]=uz;
    rho_d[tid]=rho;

    //compute fEq

    fEq[tid]=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 1 ex=1 ey=ez=0 w=1./9.
    cu=3.*(1.*ux);
    fEq[nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 2 ex=-1 ey=ez=0 w=1./9.
    cu=3.*((-1.)*ux);
    fEq[2*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 3 ex=0 ey=1 ez=0 w=1./9.
    cu=3.*(1.*uy);
    fEq[3*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 4 ex=0 ey=-1 ez=0 w=1./9.
    cu=3.*(-1.*uy);
    fEq[4*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 5 ex=ey=0 ez=1 w=1./9.
    cu=3.*(1.*uz);
    fEq[5*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    
    //speed 6 ex=ey=0 ez=-1 w=1./9.
    cu=3.*(-1.*uz);
    fEq[6*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 7 ex=ey=ez=1 w=1./72.
    cu=3.*(ux+uy+uz);
    fEq[7*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 8 ex=-1 ey=ez=1 w=1./72.
    cu=3.*(-ux+uy+uz);
    fEq[8*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
  

    //speed 9 ex=1 ey=-1 ez=1 w=1./72.
    cu=3.*(ux-uy+uz);
    fEq[9*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 10 ex=-1 ey=-1 ez=1 w=1/72
    cu=3.*(-ux-uy+uz);
    fEq[10*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 11 ex=1 ey=1 ez=-1 w=1/72
    cu=3.*(ux+uy-uz);
    fEq[11*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 12 ex=-1 ey=1 ez=-1 w=1/72
    cu=3.*(-ux+uy-uz);
    fEq[12*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 13 ex=1 ey=ez=-1 w=1/72
    cu=3.*(ux-uy-uz);
    fEq[13*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 14 ex=ey=ez=-1 w=1/72
    cu=3.*(-ux-uy-uz);
    fEq[14*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    


  }
}


err_t jktFunction(int nlhs,mxArray * plhs[],int nrhs,mxArray * prhs[]){

  if(nrhs!=9)
    return err("Usage: pc_D3Q15_InteriorNodes(fIn,fEq,rho,ux,uy,uz,INL,numIN,nnodes)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_fEq = prhs[1];
  mxArray * m_rho = prhs[2];
  mxArray * m_ux = prhs[3];
  mxArray * m_uy = prhs[4];
  mxArray * m_uz = prhs[5];
  mxArray * m_INL = prhs[6];
  int numIN = mxGetScalar(prhs[7]);
  int nnodes = mxGetScalar(prhs[8]);

  float * fIn;
  float * fEq;
  float * rho;
  float * ux;
  float * uy;
  float * uz;
  int * INL;

  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&fEq,m_fEq);
  jkt_mem((void**)&rho,m_rho);
  jkt_mem((void**)&ux,m_ux);
  jkt_mem((void**)&uy,m_uy);
  jkt_mem((void**)&uz,m_uz);
  jkt_mem((void**)&INL,m_INL);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((numIN+TPB-1)/TPB,1,1);

  pc_D3Q15_IN<<<GRIDS,BLOCKS>>>(fIn,fEq,rho,ux,uy,uz,INL,numIN,nnodes);

  return errNone;

}
