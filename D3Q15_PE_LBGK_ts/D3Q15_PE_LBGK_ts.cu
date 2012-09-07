//uses Regularized Boundary conditions for Pressure on a East boundary.

#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 64

__global__ void D3Q15_PE_LBGK_ts(float * fIn, float * fOut, 
				 const int * PE_nl, const float rho_out,
				 const int numIN, const float omega,
				 const int Nx, const int Ny, 
				 const int Nz){

  int tid_nd=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid_nd<numIN){
    int tid=PE_nl[tid_nd]-1;
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
    float cu,ux,uy,uz,rho;
    int nnodes = Nx*Ny*Nz;
    //load the data into registers
    f0=fIn[tid]; f1=fIn[nnodes+tid];
    f2=fIn[2*nnodes+tid]; f3=fIn[3*nnodes+tid];
    f4=fIn[4*nnodes+tid]; f5=fIn[5*nnodes+tid];
    f6=fIn[6*nnodes+tid]; f7=fIn[7*nnodes+tid];
    f8=fIn[8*nnodes+tid]; f9=fIn[9*nnodes+tid];
    f10=fIn[10*nnodes+tid]; f11=fIn[11*nnodes+tid];
    f12=fIn[12*nnodes+tid]; f13=fIn[13*nnodes+tid];
    f14=fIn[14*nnodes+tid];
    ux=0.; uy=0.; rho=rho_out;
    uz = -1.+((2.*(f5+f7+f8+f9+f10)+(f0+f1+f2+f3+f4)))/rho;

    float fe0,fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9,fe10,fe11,fe12,fe13,fe14;
    
    float ft1,ft2,ft3,ft4,ft5,ft6,ft7,ft8,ft9,ft10,ft11,ft12,ft13,ft14;
    float w;


    //compute equilibrium based on rho, ux,uy and uz
    //speed 0 ex=ey=ez=0 w=2./9.
    
    fe0=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 1 ex=1 ey=ez=0 w=1./9.
    cu=3.*(1.*ux);
    fe1=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		     1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 2 ex=-1 ey=ez=0 w=1./9.
    cu=3.*((-1.)*ux);
    fe2=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		     1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 3 ex=0 ey=1 ez=0 w=1./9.
    cu=3.*(1.*uy);
    fe3=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		     1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 4 ex=0 ey=-1 ez=0 w=1./9.
    cu=3.*(-1.*uy);
    fe4=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		     1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 5 ex=ey=0 ez=1 w=1./9.
    cu=3.*(1.*uz);
    fe5=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		     1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 6 ex=ey=0 ez=-1 w=1./9.
    cu=3.*(-1.*uz);
    fe6=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
    		     1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 7 ex=ey=ez=1 w=1./72.
    cu=3.*(ux+uy+uz);
    fe7=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		      1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 8 ex=-1 ey=ez=1 w=1./72.
    cu=3.*(-ux+uy+uz);
    fe8=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		      1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 9 ex=1 ey=-1 ez=1 w=1./72.
    cu=3.*(ux-uy+uz);
    fe9=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		      1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 10 ex=-1 ey=-1 ez=1 w=1/72
    cu=3.*(-ux-uy+uz);
    fe10=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 11 ex=1 ey=1 ez=-1 w=1/72
    cu=3.*(ux+uy-uz);
    fe11=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
    

    //speed 12 ex=-1 ey=1 ez=-1 w=1/72
    cu=3.*(-ux+uy-uz);
    fe12=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
  

    //speed 13 ex=1 ey=ez=-1 w=1/72
    cu=3.*(ux-uy-uz);
    fe13=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));
   

    //speed 14 ex=ey=ez=-1 w=1/72
    cu=3.*(-ux-uy-uz);
    fe14=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
    		       1.5*(ux*ux+uy*uy+uz*uz));


    f6=fe6+(f5-fe5); 
    f11=fe11+(f10-fe10); 
    f12=fe12+(f9-fe9); 
    f13=fe13+(f8-fe8); 
    f14=fe14+(f7-fe7); 

    ft1=f1-fe1; 
    ft2=f2-fe2;
    ft3=f3-fe3;
    ft4=f4-fe4;
    ft5=f5-fe5;
    ft6=f6-fe6;
    ft7=f7-fe7;
    ft8=f8-fe8;
    ft9=f9-fe9;
    ft10=f10-fe10;
    ft11=f11-fe11;
    ft12=f12-fe12;
    ft13=f13-fe13;
    ft14=f14-fe14;

    //now, multiply by f# = ((ft#)*Q_flat)*Q_flat'
    f0=0;
    f1=ft1+ft2+ft7+ft8+ft9+ft10+ft11+ft12+ft13+ft14;
    f2=f1;
    f3=ft3+ft4+ft7+ft8+ft9+ft10+ft11+ft12+ft13+ft14;
    f4=f3;
    f5=ft5+ft6+ft7+ft8+ft9+ft10+ft11+ft12+ft13+ft14;
    f6=f5;
    f7=ft1+ft2+ft3+ft4+ft5+ft6+9.*ft7+ft8+ft9+ft10+ft11+ft12+ft13+9.*ft14;
    f8=ft1+ft2+ft3+ft4+ft5+ft6+ft7+9.*ft8+ft9+ft10+ft11+ft12+9.*ft13+ft14;
    f9=ft1+ft2+ft3+ft4+ft5+ft6+ft7+ft8+9.*ft9+ft10+ft11+9.*ft12+ft13+ft14;
    f10=ft1+ft2+ft3+ft4+ft5+ft6+ft7+ft8+ft9+9.*ft10+9.*ft11+ft12+ft13+ft14;
    f11=ft1+ft2+ft3+ft4+ft5+ft6+ft7+ft8+ft9+9.*ft10+9.*ft11+ft12+ft13+ft14;
    f12=ft1+ft2+ft3+ft4+ft5+ft6+ft7+ft8+9.*ft9+ft10+ft11+9.*ft12+ft13+ft14;
    f13=ft1+ft2+ft3+ft4+ft5+ft6+ft7+9.*ft8+ft9+ft10+ft11+ft12+9.*ft13+ft14;
    f14=ft1+ft2+ft3+ft4+ft5+ft6+9.*ft7+ft8+ft9+ft10+ft11+ft12+ft13+9.*ft14;

    //update fIn for all velocities based on strain tensor
    //f0, still equals 0..
    cu = 9./2.; w = 1./9.;

    //fIn[..] = fe#+f#
    f0=fe0;

    f1=fe1+f1*(cu)*w;
    f2=fe2+f2*(cu)*w;
    f3=fe3+f3*cu*w;
    f4=fe4+f4*cu*w;
    f5=fe5+f5*cu*w;
    f6=fe6+f6*cu*w;
    w = 1./72.;
    f7=fe7+f7*cu*w;
    f8=fe8+f8*cu*w;
    f9=fe9+f9*cu*w;
    f10=fe10+f10*cu*w;
    f11=fe11+f11*cu*w;
    f12=fe12+f12*cu*w;
    f13=fe13+f13*cu*w;
    f14=fe14+f14*cu*w;





    //now, relax with the new fIn and previously computed fEq
    f0=f0-omega*(f0-fe0);
    f1=f1-omega*(f1-fe1);
    f2=f2-omega*(f2-fe2);
    f3=f3-omega*(f3-fe3);
    f4=f4-omega*(f4-fe4);
    f5=f5-omega*(f5-fe5);
    f6=f6-omega*(f6-fe6);
    f7=f7-omega*(f7-fe7);
    f8=f8-omega*(f8-fe8);
    f9=f9-omega*(f9-fe9);
    f10=f10-omega*(f10-fe10);
    f11=f11-omega*(f11-fe11);
    f12=f12-omega*(f12-fe12);
    f13=f13-omega*(f13-fe13);
    f14=f14-omega*(f14-fe14);

    //now stream in all directions except to the east (positive Z direction)
    //due to how boundary is defined, no need to check for periodic logic.

    //now, everybody streams...
    int X_t, Y_t, Z_t;
    int tid_t;

    int Z = tid/(Nx*Ny);
    int Y = (tid - Z*Nx*Ny)/Nx;
    int X = tid - Z*Nx*Ny - Y*Nx;
    //speed 0 ex=ey=ez=0
    fOut[tid]=f0;

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

  if(nrhs!=9)
    return err("Usage: D3Q15_PE_LBGK_ts(fIn,fOut,PE_nl,rho_out,numIN,omega,Nx,Ny,Nz");

  mxArray * m_fIn = prhs[0];
  mxArray * m_fOut = prhs[1];
  mxArray * m_PE_nl = prhs[2];
  float rho_out = mxGetScalar(prhs[3]);
  int numIN=mxGetScalar(prhs[4]);
  float omega = mxGetScalar(prhs[5]);
  int Nx=mxGetScalar(prhs[6]);
  int Ny=mxGetScalar(prhs[7]);
  int Nz=mxGetScalar(prhs[8]);

  float * fIn;
  float * fOut;
  int * PE_nl;
  

  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&fOut,m_fOut);
  jkt_mem((void**)&PE_nl,m_PE_nl);
 

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((numIN+TPB-1)/TPB,1,1);

  D3Q15_PE_LBGK_ts<<<GRIDS,BLOCKS>>>(fIn,fOut,PE_nl,rho_out,numIN,omega,Nx,Ny,Nz);


  return errNone;

}
