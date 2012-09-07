#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 128

__global__ void D3Q15_IntNode_LBGK_ts(float * fIn, float * fOut, int * INL, 
				      const int numIN, const float omega,
				      const int Nx, const int Ny, const int Nz){

  int tid_nd = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid_nd<numIN){
    int tid=INL[tid_nd]-1;
    int nnodes = Nx*Ny*Nz;
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

    //relax
    //speed 0 ex=ey=ez=0 w=2./9.
    float fEq;
    fEq=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
    f0=f0-omega*(f0-fEq);

    //speed 1 ex=1 ey=ez=0 w=1./9.
    cu=3.*(1.*ux);
    fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    f1=f1-omega*(f1-fEq);

    //speed 2 ex=-1 ey=ez=0 w=1./9.
    cu=3.*((-1.)*ux);
    fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    f2=f2-omega*(f2-fEq);

    //speed 3 ex=0 ey=1 ez=0 w=1./9.
    cu=3.*(1.*uy);
    fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    f3=f3-omega*(f3-fEq);

    //speed 4 ex=0 ey=-1 ez=0 w=1./9.
    cu=3.*(-1.*uy);
    fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    f4=f4-omega*(f4-fEq);

    //speed 5 ex=ey=0 ez=1 w=1./9.
    cu=3.*(1.*uz);
    fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    f5=f5-omega*(f5-fEq);

    //speed 6 ex=ey=0 ez=-1 w=1./9.
    cu=3.*(-1.*uz);
    fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		     1.5*(ux*ux+uy*uy+uz*uz));
    f6=f6-omega*(f6-fEq);

    //speed 7 ex=ey=ez=1 w=1./72.
    cu=3.*(ux+uy+uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f7=f7-omega*(f7-fEq);

    //speed 8 ex=-1 ey=ez=1 w=1./72.
    cu=3.*(-ux+uy+uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f8=f8-omega*(f8-fEq);

    //speed 9 ex=1 ey=-1 ez=1 w=1./72.
    cu=3.*(ux-uy+uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f9=f9-omega*(f9-fEq);

    //speed 10 ex=-1 ey=-1 ez=1 w=1/72
    cu=3.*(-ux-uy+uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f10=f10-omega*(f10-fEq);

    //speed 11 ex=1 ey=1 ez=-1 w=1/72
    cu=3.*(ux+uy-uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f11=f11-omega*(f11-fEq);

    //speed 12 ex=-1 ey=1 ez=-1 w=1/72
    cu=3.*(-ux+uy-uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f12=f12-omega*(f12-fEq);

    //speed 13 ex=1 ey=ez=-1 w=1/72
    cu=3.*(ux-uy-uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f13=f13-omega*(f13-fEq);

    //speed 14 ex=ey=ez=-1 w=1/72
    cu=3.*(-ux-uy-uz);
    fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
		      1.5*(ux*ux+uy*uy+uz*uz));
    f14=f14-omega*(f14-fEq);

    //stream...
    //in this case, since it's an interior node, I can safely assume
    //that I will not need to apply periodic streaming logic.
    int Z = tid/(Nx*Ny);
    int Y = (tid-Z*Nx*Ny)/Nx;
    int X = (tid-Z*Nx*Ny-Y*Nx);

 //now, everybody streams...
    int X_t, Y_t, Z_t;
    int tid_t;

    
    //speed 1 ex=1 ey=ez=0
    X_t=X+1; Y_t=Y; Z_t=Z;
    if(X_t==Nx) X_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[Nx*Ny*Nz+tid_t]=f1;

    //speed 2 ex=-1 ey=ez=0;
    X_t=X-1; Y_t=Y; Z_t=Z;
    if(X_t<0) X_t=(Nx-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[2*Nx*Ny*Nz+tid_t]=f2;

    //speed 3 ex=0 ey=1 ez=0
    X_t=X; Y_t=Y+1; Z_t=Z;
    if(Y_t==Ny) Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[3*Nx*Ny*Nz+tid_t]=f3;

    //speed 4 ex=0 ey=-1 ez=0
    X_t=X; Y_t=Y-1; Z_t=Z;
    if(Y_t<0) Y_t=(Ny-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[4*Nx*Ny*Nz+tid_t]=f4;

    //speed 5 ex=ey=0 ez=1
    X_t=X; Y_t=Y; Z_t=Z+1;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[5*Nx*Ny*Nz+tid_t]=f5;

    //speed 6 ex=ey=0 ez=-1
    X_t=X; Y_t=Y; Z_t=Z-1;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[6*Nx*Ny*Nz+tid_t]=f6;

    //speed 7 ex=ey=ez=1
    X_t=X+1; Y_t=Y+1; Z_t=Z+1;
    if(X_t==Nx) X_t=0;
    if(Y_t==Ny) Y_t=0;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[7*Nx*Ny*Nz+tid_t]=f7;

    //speed 8 ex=-1 ey=1 ez=1
    X_t=X-1; Y_t=Y+1; Z_t=Z+1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t==Ny) Y_t=0;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[8*Nx*Ny*Nz+tid_t]=f8;

    //speed 9 ex=1 ey=-1 ez=1
    X_t=X+1; Y_t=Y-1; Z_t=Z+1;
    if(X_t==Nx) X_t=0;
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[9*Nx*Ny*Nz+tid_t]=f9;

    //speed 10 ex=-1 ey=-1 ez=1
    X_t=X-1; Y_t=Y-1; Z_t=Z+1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[10*Nx*Ny*Nz+tid_t]=f10;

    //speed 11 ex=1 ey=1 ez=-1
    X_t=X+1; Y_t=Y+1; Z_t=Z-1;
    if(X_t==Nx) X_t=0;
    if(Y_t==Ny) Y_t=0;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[11*Nx*Ny*Nz+tid_t]=f11;

    //speed 12 ex=-1 ey=1 ez=-1
    X_t=X-1; Y_t=Y+1; Z_t=Z-1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t==Ny) Y_t=0;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[12*Nx*Ny*Nz+tid_t]=f12;

    //speed 13 ex=1 ey=-1 ez=-1
    X_t=X+1; Y_t=Y-1; Z_t=Z-1;
    if(X_t==Nx) X_t=0;
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[13*Nx*Ny*Nz+tid_t]=f13;

    //speed 14 ex=ey=ez=-1
    X_t=X-1; Y_t=Y-1; Z_t=Z-1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[14*Nx*Ny*Nz+tid_t]=f14;



  }
}


err_t jktFunction(int nlhs,mxArray * plhs[],int nrhs,mxArray * prhs[]){

  if(nrhs!=8)
    return err("Usage: D3Q15_InteriorNode_LBGK_ts(fIn,fOut,INL,numIN,omega,Nx,Ny,Nz)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_fOut = prhs[1];
  mxArray * m_INL = prhs[2];
  int numIN = mxGetScalar(prhs[3]);
  float omega = mxGetScalar(prhs[4]);
  int Nx = mxGetScalar(prhs[5]);
  int Ny = mxGetScalar(prhs[6]);
  int Nz = mxGetScalar(prhs[7]);

  float * fIn;
  float * fOut;
  int * INL;

  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&fOut,m_fOut);
  jkt_mem((void**)&INL,m_INL);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((numIN+TPB-1)/TPB,1,1);

  D3Q15_IntNode_LBGK_ts<<<GRIDS,BLOCKS>>>(fIn,fOut,INL,numIN,omega,Nx,Ny,Nz);

  return errNone;

}
