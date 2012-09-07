#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 128

__global__ void D3Q15_resetToFeq(float * fOut,  float * fIn,int * snl,
				 int * vel_nl, float * u_p, 
				 float *v_p,
				 float * w_p,const int * ndList,
				 const int numList,const int nnodes){
  // int X=threadIdx.x+blockIdx.x*blockDim.x;
  // int Y=threadIdx.y+blockIdx.y*blockDim.y;
  // int Z=threadIdx.z+blockIdx.z*blockDim.z;

  //need to determine X,Y and Z based on the node number...
  int tid_nd = threadIdx.x+blockIdx.x*blockDim.x;

  if(tid_nd<numList){
    //int tid=X+Y*Nx+Z*Nx*Ny;
    int tid = ndList[tid_nd]-1;//<-- since ndList gives the tid number as 1-base
  
    
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

    //compute density
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy/=rho;
    float uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz/=rho;

   


    if((vel_nl[tid]==1)||(snl[tid]==1)){
     
      ux=u_p[tid]; uy=v_p[tid]; uz=w_p[tid];

    }
    
    //reset to fEq      

      //speed 0 ex=ey=ez=0 w=2./9.
      float fEq;
      fEq=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
      //f0=f0-omega*(f0-fEq);
      fOut[tid]=fEq;

      //speed 1 ex=1 ey=ez=0 w=1./9.
      cu=3.*(1.*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f1=f1-omega*(f1-fEq);
      fOut[nnodes+tid]=fEq;

      //speed 2 ex=-1 ey=ez=0 w=1./9.
      cu=3.*((-1.)*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f2=f2-omega*(f2-fEq);
      fOut[2*nnodes+tid]=fEq;

      //speed 3 ex=0 ey=1 ez=0 w=1./9.
      cu=3.*(1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f3=f3-omega*(f3-fEq);
      fOut[3*nnodes+tid]=fEq;

      //speed 4 ex=0 ey=-1 ez=0 w=1./9.
      cu=3.*(-1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f4=f4-omega*(f4-fEq);
      fOut[4*nnodes+tid]=fEq;

      //speed 5 ex=ey=0 ez=1 w=1./9.
      cu=3.*(1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      // f5=f5-omega*(f5-fEq);
      fOut[5*nnodes+tid]=fEq;

      //speed 6 ex=ey=0 ez=-1 w=1./9.
      cu=3.*(-1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f6=f6-omega*(f6-fEq);
      fOut[6*nnodes+tid]=fEq;

      //speed 7 ex=ey=ez=1 w=1./72.
      cu=3.*(ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f7=f7-omega*(f7-fEq);
      fOut[7*nnodes+tid]=fEq;

      //speed 8 ex=-1 ey=ez=1 w=1./72.
      cu=3.*(-ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f8=f8-omega*(f8-fEq);
      fOut[8*nnodes+tid]=fEq;

      //speed 9 ex=1 ey=-1 ez=1 w=1./72.
      cu=3.*(ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f9=f9-omega*(f9-fEq);
      fOut[9*nnodes+tid]=fEq;

      //speed 10 ex=-1 ey=-1 ez=1 w=1/72
      cu=3.*(-ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f10=f10-omega*(f10-fEq);
      fOut[10*nnodes+tid]=fEq;

      //speed 11 ex=1 ey=1 ez=-1 w=1/72
      cu=3.*(ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f11=f11-omega*(f11-fEq);
      fOut[11*nnodes+tid]=fEq;

      //speed 12 ex=-1 ey=1 ez=-1 w=1/72
      cu=3.*(-ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f12=f12-omega*(f12-fEq);
      fOut[12*nnodes+tid]=fEq;

      //speed 13 ex=1 ey=ez=-1 w=1/72
      cu=3.*(ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f13=f13-omega*(f13-fEq);
      fOut[13*nnodes+tid]=fEq;

      //speed 14 ex=ey=ez=-1 w=1/72
      cu=3.*(-ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f14=f14-omega*(f14-fEq);
      fOut[14*nnodes+tid]=0;


   
  

  }
}




err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){

  if(nrhs!=10)
    return err("Usage: D3Q15_resetToFeq(fOut,fIn,snl,vel_nl,u_p,v_p,w_p,nodeList,numList,nnodes)");


  mxArray * m_fOut = prhs[0];
  mxArray * m_fIn=prhs[1];
  mxArray * m_snl=prhs[2];
  mxArray * m_vel_nl=prhs[3];
  mxArray * m_u_p=prhs[4];
  mxArray * m_v_p=prhs[5];
  mxArray * m_w_p=prhs[6];
 
  mxArray * m_ndList=prhs[7];
  int numList = mxGetScalar(prhs[8]);
  int nnodes=mxGetScalar(prhs[9]);

 

  float * fOut_d;
  float * fIn_d;
  int * snl_d;
  int * vel_nl_d;
  float * u_p_d;
  float * v_p_d;
  float * w_p_d;
  int * ndList;

  jkt_mem((void**)&fOut_d,m_fOut);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&snl_d,m_snl);
  jkt_mem((void**)&vel_nl_d,m_vel_nl);
  jkt_mem((void**)&u_p_d,m_u_p);
  jkt_mem((void**)&v_p_d,m_v_p);
  jkt_mem((void**)&w_p_d,m_w_p);
  jkt_mem((void**)&ndList,m_ndList);
  

 
 
  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  D3Q15_resetToFeq<<<GRIDS,BLOCKS>>>(fOut_d,fIn_d,snl_d,vel_nl_d,
				     u_p_d,v_p_d,w_p_d,ndList,numList,nnodes);


  return errNone;

}
