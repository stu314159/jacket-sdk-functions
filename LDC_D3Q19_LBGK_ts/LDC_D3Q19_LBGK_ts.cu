#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 128

__global__ void ldc_D3Q19_LBGK_ts(float * fOut, float * fIn, float * U,
				  const float u_bc, const float omega,
				  const int Nx, const int Ny,
				  const int Nz){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes=Nx*Ny*Nz;
  if(tid<(nnodes)){
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18;
    float cu;
    float w;
    //load the data into the registers
    f0=fIn[tid]; f1=fIn[nnodes+tid];
    f2=fIn[2*nnodes+tid]; f3=fIn[3*nnodes+tid];
    f4=fIn[4*nnodes+tid]; f5=fIn[5*nnodes+tid];
    f6=fIn[6*nnodes+tid]; f7=fIn[7*nnodes+tid];
    f8=fIn[8*nnodes+tid]; f9=fIn[9*nnodes+tid];
    f10=fIn[10*nnodes+tid]; f11=fIn[11*nnodes+tid];
    f12=fIn[12*nnodes+tid]; f13=fIn[13*nnodes+tid];
    f14=fIn[14*nnodes+tid]; f15=fIn[15*nnodes+tid];
    f16=fIn[16*nnodes+tid]; f17=fIn[17*nnodes+tid];
    f18=fIn[18*nnodes+tid];
    //compute density and velocity
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f15-f16+f17-f18; uy/=rho;
    float uz=f5-f6+f11+f12-f13-f14+f15+f16-f17-f18; uz/=rho;

    int Z = tid/(Nx*Ny);
    int Y = (tid - Z*Nx*Ny)/Nx;
    int X = tid - Z*Nx*Ny - Y*Nx;

    if((X==0)&&(!((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))))){
      //apply velocity boundary condition here...
      //u_bc is the prescribed y velocity

      //speed 1 (1,0,0) w=1./18.
      w = 1./18.;
      cu = 3.*(-ux);
      f1+=w*rho*cu;

      //speed 2 (-1,0,0) 
      cu=3.*(-1.)*(-ux);
      f2+=w*rho*cu;

      //speed 3 (0,1,0)
      cu = 3.*(u_bc-uy);
      f3+=w*rho*cu;

      //speed 4 (0,-1,0)
      cu = 3.*(-1.)*(u_bc-uy);
      f4+=w*rho*cu;

      //speed 5 (0,0,1)
      cu = 3.*(-uz);
      f5+=w*rho*cu;

      //speed 6 (0,0,-1)
      cu = 3.*(-1.)*(-uz);
      f6+=w*rho*cu;

      w = 1./36.;
      //speed 7 (1,1,0)
      cu = 3.*((-ux)+(u_bc-uy));
      f7+=w*rho*cu;

      //speed 8 ( -1,1,0)
      cu = 3.*((-1.)*(-ux) + (u_bc-uy));
      f8+=w*rho*cu;

      //speed 9 (1,-1,0)
      cu = 3.*((-ux) -(u_bc-uy));
      f9+=w*rho*cu;

      //speed 10 (-1,-1,0)
      cu = 3.*(-(-ux) -(u_bc-uy));
      f10+=w*rho*cu;

      //speed 11 (1,0,1)
      cu = 3.*((-ux)+(-uz));
      f11+=w*rho*cu;

      //speed 12 (-1,0,1)
      cu = 3.*(ux -uz);
      f12+=w*rho*cu;

      //speed 13 (1,0,-1)
      cu = 3.*(-ux + uz);
      f13+=w*rho*cu;

      //speed 14 (-1,0,-1)
      cu = 3.*(ux+uz);
      f14+=w*rho*cu;

      //speed 15 ( 0,1,1)
      cu = 3.*((u_bc-uy)-uz);
      f15+=w*rho*cu;

      //speed 16 (0,-1,1)
      cu = 3.*(-(u_bc-uy)-uz);
      f16+=w*rho*cu;

      //speed 17 (0,1,-1)
      cu = 3.*((u_bc-uy)+uz);
      f17+=w*rho*cu;

      //speed 18 (0,-1,-1)
      cu = 3.*((uy-u_bc)+uz);
      f18+=w*rho*cu;

      ux=0.; uy =u_bc; uz = 0.;

    }//if(lnl[tid]==1)...

   
    //if(snl[tid]==1){
    if(((Y==0)||(Y==(Ny-1))||(Z==0)||(Z==(Nz-1))||(X==(Nx-1)))){
      //bounce back
      ux=0.; uy=0.; uz=0.;

      
      //1 -- 2
      cu=f1;f1=f2;f2=cu;
      // 3 -- 4
      cu=f3;f3=f4;f4=cu;
      //5--6
      cu=f5;f5=f6;f6=cu;
      //7--10
      cu=f7;f7=f10;f10=cu;
      //8--9
      cu=f8;f8=f9;f9=cu;
      //11-14
      cu=f11;f11=f14;f14=cu;
      //12-13
      cu=f12;f12=f13;f13=cu;
      //15-18
      cu=f15;f15=f18;f18=cu;
      //16-17
      cu=f16;f16=f17;f17=cu;

    }


    //relax
     
    float fe0,fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9,fe10,fe11,fe12,fe13,fe14,fe15,fe16,fe17,fe18;

    //speed 0, ex=ey=ez=0, w=1/3
    fe0=rho*(1./3.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
   
    //speed 1, ex=1, ey=ez=0, w=1/18
    cu = 3.*(1.*ux);
    fe1=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 2, ex=-1, ey=ez=0
    cu=3.*(-1.*ux);
    fe2=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 3 (0,1,0)
    cu=3.*(uy);
 
    fe3=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 4 (0,-1,0)
    cu = 3.*(-uy);
    fe4=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 5 (0,0,1)
    cu = 3.*(uz);
    fe5=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 6 (0,0,-1)
    cu = 3.*(-uz);
    fe6=rho*(1./18.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 7 (1,1,0)  w= 1/36
    cu = 3.*(ux+uy);
    fe7=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 8 (-1,1,0)
    cu = 3.*(-ux+uy);
    fe8=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 9 (1,-1,0)
    cu=3.*(ux-uy);
    fe9=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 10 (-1,-1,0)
    cu = 3.*(-ux-uy);
    fe10=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 11 (1,0,1)
    cu = 3.*(ux+uz);
    fe11=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 12 (-1,0,1)
    cu = 3.*(-ux+uz);
    fe12=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 13 (1,0,-1)
    cu = 3.*(ux-uz);
    fe13=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 14 (-1,0,-1)
    cu=3.*(-ux-uz);
    fe14=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 15 (0,1,1)
    cu=3.*(uy+uz);
    fe15=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 16 (0,-1,1)
    cu=3.*(-uy+uz);
    fe16=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 17 (0,1,-1)
    cu=3.*(uy-uz);
    fe17=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));
    //speed 18 (0,-1,-1)
    cu=3.*(-uy-uz);
    fe18=rho*(1./36.)*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux+uy*uy+uz*uz));



    //everyone relaxes towards equilibrium
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
    f15=f15-omega*(f15-fe15);
    f16=f16-omega*(f16-fe16);
    f17=f17-omega*(f17-fe17);
    f18=f18-omega*(f18-fe18);



   

    U[tid]=sqrt(ux*ux+uy*uy+uz*uz);

    //now, streaming...
    int X_t,Y_t,Z_t,tid_t;

    //speed 0 (0,0,0)
    fOut[tid]=f0;
    //stream(fOut,f0,0,X,Y,Z,0,0,0,Nx,Ny,Nz);

    //speed 1 (1,0,0)
    X_t=X+1;Y_t=Y; Z_t=Z;
    if(X_t==Nx) X_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[nnodes+tid_t]=f1;
    
    //speed 2 (-1,0,0)
    X_t=X-1; Y_t=Y; Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[2*nnodes+tid_t]=f2;

    //speed 3 (0,1,0)
    X_t=X; Y_t=Y+1; Z_t=Z;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    //tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[3*nnodes+tid_t]=f3;
    //speed 4 ( 0,-1,0)
    X_t=X; Y_t=Y-1; Z_t=Z;
    if(Y_t<0)Y_t=Ny-1;

    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[4*nnodes+tid_t]=f4; 
    //speed 5 ( 0,0,1)
    X_t=X;Y_t=Y;Z_t=Z+1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[5*nnodes+tid_t]=f5;
    //speed 6 (0,0,-1)
    X_t=X; Y_t=Y;Z_t=Z-1;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[6*nnodes+tid_t]=f6;
    //speed 7 (1,1,0)
    X_t=X+1;Y_t=Y+1;Z_t=Z;
    if(X_t==Nx)X_t=0;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[7*nnodes+tid_t]=f7;
    //speed 8 (-1,1,0)
    X_t=X-1;Y_t=Y+1;Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    if(Y_t==Ny)Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[8*nnodes+tid_t]=f8;
    //speed 9 (1,-1,0)
    X_t=X+1;Y_t=Y-1;Z_t=Z;
    if(X_t==Nx)X_t=0;
    if(Y_t<0)Y_t=Ny-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[9*nnodes+tid_t]=f9;
    //speed 10 (-1,-1,0)
    X_t=X-1;Y_t=Y-1;Z_t=Z;
    if(X_t<0)X_t=Nx-1;
    if(Y_t<0)Y_t=Ny-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[10*nnodes+tid_t]=f10;
    //speed 11 (1,0,1)
    X_t=X+1;Y_t=Y;Z_t=Z+1;
    if(X_t==Nx)X_t=0;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[11*nnodes+tid_t]=f11;
    //speed 12 (-1,0,1)
    X_t=X-1;Y_t=Y;Z_t=Z+1;
    if(X_t<0)X_t=Nx-1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[12*nnodes+tid_t]=f12;
    //speed 13 (1,0,-1)
    X_t=X+1;Y_t=Y;Z_t=Z-1;
    if(X_t==Nx)X_t=0;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[13*nnodes+tid_t]=f13;
    //speed 14 (-1,0,-1)
    X_t=X-1;Y_t=Y;Z_t=Z-1;
    if(X_t<0)X_t=Nx-1;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[14*nnodes+tid_t]=f14;
    //speed 15 (0,1,1)
    X_t=X;Y_t=Y+1;Z_t=Z+1;
    if(Y_t==Ny)Y_t=0;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[15*nnodes+tid_t]=f15;
    //speed 16 (0,-1,1)
    X_t=X;Y_t=Y-1;Z_t=Z+1;
    if(Y_t<0)Y_t=Ny-1;
    if(Z_t==Nz)Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[16*nnodes+tid_t]=f16;

    //speed 17 (0,1,-1)
    X_t=X;Y_t=Y+1;Z_t=Z-1;
    if(Y_t==Ny)Y_t=0;
    if(Z_t<0)Z_t=Nz-1;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[17*nnodes+tid_t]=f17;


    //speed 18 ( 0,-1,-1)
    X_t=X;Y_t=Y-1;Z_t=Z-1;
    if(Y_t<0)Y_t=Ny-1;
    if(Z_t<0)Z_t=Nz-1;

    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[18*nnodes+tid_t]=f18;
  }
}


err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){

  if(nrhs!=8)
    return err("Usage: LDC_D3Q19_LBGK_ts(fOut,fIn,U,u_bc,omega,Nx,Ny,Nz)");


  mxArray * m_fIn = prhs[0];
  mxArray * m_fOut=prhs[1];
  mxArray * m_U = prhs[2];
  float u_bc = mxGetScalar(prhs[3]);
  float omega = mxGetScalar(prhs[4]);
  int Nx = mxGetScalar(prhs[5]);
  int Ny = mxGetScalar(prhs[6]);
  int Nz = mxGetScalar(prhs[7]);

 

  float * fOut_d;
  float * fIn_d;
  float * U_d;

  jkt_mem((void**)&fOut_d,m_fOut);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&U_d,m_U);

 

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((Nx*Ny*Nz+TPB-1)/TPB,1,1);

  ldc_D3Q19_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut_d,fIn_d,U_d,u_bc,omega,Nx,Ny,Nz);


  return errNone;

}
