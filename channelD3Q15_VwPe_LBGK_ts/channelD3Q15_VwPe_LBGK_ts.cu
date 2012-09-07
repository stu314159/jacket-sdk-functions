#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

// Right now, this doesn't seem to be working.



#define TPB 92

__global__ void channelD3Q15_VwPe_LBGK_ts(float * fOut,  float * fIn,int * inl,
					  int * onl,int * snl,float * uz_p,
					  const float rho_out,
					  const float nu_lbm,
					  const int Nx, const int Ny,
					  const int Nz){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;

  if(tid<(Nx*Ny*Nz)){
    int X,Y,Z;
    Z = (tid)/(Nx*Ny);
    Y = (tid-Z*Nx*Ny)/Nx;
    X = tid - Z*Nx*Ny - Y*Nx;

    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
    float cu;
    //load the data into registers
    f0=fIn[tid]; f1=fIn[Nx*Ny*Nz+tid];
    f2=fIn[2*Nx*Ny*Nz+tid]; f3=fIn[3*Nx*Ny*Nz+tid];
    f4=fIn[4*Nx*Ny*Nz+tid]; f5=fIn[5*Nx*Ny*Nz+tid];
    f6=fIn[6*Nx*Ny*Nz+tid]; f7=fIn[7*Nx*Ny*Nz+tid];
    f8=fIn[8*Nx*Ny*Nz+tid]; f9=fIn[9*Nx*Ny*Nz+tid];
    f10=fIn[10*Nx*Ny*Nz+tid]; f11=fIn[11*Nx*Ny*Nz+tid];
    f12=fIn[12*Nx*Ny*Nz+tid]; f13=fIn[13*Nx*Ny*Nz+tid];
    f14=fIn[14*Nx*Ny*Nz+tid];

    //compute density
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy/=rho;
    float uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz/=rho;

    


    if(onl[tid]==1){
      rho = rho_out; ux=0; uy=0;
      uz = -1.+(2.*(f5+f7+f8+f9+f10)+(f0+f1+f2+f3+f4))/rho;

      f6=f5-(2./3.)*rho*uz;
      f11=f10-(1./12.)*rho*uz-0.25*((f1-f2)+(f3-f4));
      f12=f9-(1./12.)*rho*uz-0.25*(-(f1-f2)+(f3-f4));
      f13=f8-(1./12.)*rho*uz-0.25*((f1-f2)-(f3-f4));
      f14=f7-(1./12.)*rho*uz-0.25*(-(f1-f2)-(f3-f4));      



    }
    if(inl[tid]==1){
      //Zou/He prescribed velocity at X/Y plane at lower Z (called "west" in 3D)
      uz = uz_p[tid]; ux=0; uy=0;
      rho = (1./(1.+uz))*(2.*(f6+f11+f12+f13+f14) + 
			  (f0+f1+f2+f3+f4));

      f5=f6+(2./3.)*rho*uz;
      f7=f14+(1./12.)*rho*uz-0.25*((f1-f2)+(f3-f4));
      f8=f13+(1./12.)*rho*uz-0.25*(-(f1-f2)+(f3-f4));
      f9=f12+(1./12.)*rho*uz-0.25*((f1-f2)-(f3-f4));
      f10=f11+(1./12.)*rho*uz-0.25*(-(f1-f2)-(f3-f4));



    }

    if(snl[tid]==1){

      // 1--2
      cu=f2; f2=f1; f1=cu;
      //3--4
      cu=f4; f4=f3; f3=cu;
      //5--6
      cu=f6; f6=f5; f5=cu;
      //7--14
      cu=f14; f14=f7; f7=cu;
      //8--13
      cu=f13; f13=f8; f8=cu;
      //9--12
      cu=f12; f12=f9; f9=cu;
      //10--11
      cu=f11; f11=f10; f10=cu;


    }else{
      //relax
      //speed 0 ex=ey=ez=0 w=2./9.
      float fEq;
      float omega = 1./(3.*nu_lbm+0.5);
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


    }

    //now, everybody streams...
    int X_t, Y_t, Z_t;
    int tid_t;

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




err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){

  if(nrhs!=11)
    return err("Usage: channelD3Q15_VwPe_LBGK_ts(fOut,fIn,inl,onl,snl,uz_p,rho_out,nu_lbm,Nx,Ny,Nz)");


  mxArray * m_fOut = prhs[0];
  mxArray * m_fIn=prhs[1];
  mxArray * m_inl=prhs[2];
  mxArray * m_onl=prhs[3];
  mxArray * m_snl=prhs[4];
  mxArray * m_uz_p=prhs[5];
  float rho_out =mxGetScalar(prhs[6]);
  float nu_lbm = mxGetScalar(prhs[7]);
  int Nx = mxGetScalar(prhs[8]);
  int Ny = mxGetScalar(prhs[9]);
  int Nz = mxGetScalar(prhs[10]);

 

  float * fOut_d;
  float * fIn_d;
  int * snl_d;
  int * inl_d;
  int * onl_d;
  float * uz_p;
  

  jkt_mem((void**)&fOut_d,m_fOut);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&snl_d,m_snl);
  jkt_mem((void**)&inl_d,m_inl);
  jkt_mem((void**)&onl_d,m_onl);
  jkt_mem((void**)&uz_p,m_uz_p);
  
  

  int nnodes = Nx*Ny*Nz;

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  channelD3Q15_VwPe_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut_d,fIn_d,inl_d,onl_d,
					      snl_d,uz_p,rho_out,nu_lbm,
					      Nx,Ny,Nz);


  return errNone;

}
