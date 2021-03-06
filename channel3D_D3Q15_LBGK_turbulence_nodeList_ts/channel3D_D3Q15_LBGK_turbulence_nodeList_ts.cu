#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 128

// device function to compute t_s given nu and lsn (local stress norm - which is the Frobinius norm of the local stress tensor...at least I *think* that's what it is...)

__device__ float get_omega_turb(float nu, float lsn){

  //the Smagorinski constant (for this version) will be fixed at 0.03
  const float C_sq = 0.03*0.03;
  float tau_s;
  tau_s = (1./(6.0*C_sq));
  float tmp_f = nu*nu + 18.*C_sq*lsn;
  tmp_f = sqrt(tmp_f);
  tau_s = tau_s*tmp_f - nu;
  tau_s = 3.0*(nu+C_sq*tau_s)+0.5;
  return 1./tau_s;


}


__global__ void channel3D_D3Q15_LBGK_turb_NL_ts(float * fOut,  float * fIn,int * snl,
					   int * vel_nl, float * u_p, 
					   float *v_p,
					   float * w_p, const float nu,
					   const int Nx, const int Ny,
					   const int Nz,const int * ndList,
					   const int numList){
  // int X=threadIdx.x+blockIdx.x*blockDim.x;
  // int Y=threadIdx.y+blockIdx.y*blockDim.y;
  // int Z=threadIdx.z+blockIdx.z*blockDim.z;

  //need to determine X,Y and Z based on the node number...
  int tid_nd = threadIdx.x+blockIdx.x*blockDim.x;

  if(tid_nd<numList){
    //int tid=X+Y*Nx+Z*Nx*Ny;
    int tid = ndList[tid_nd]-1;//<-- since ndList gives the tid number as 1-base
    int Z = tid/(Nx*Ny);
    int Y = (tid - (Z*Nx*Ny))/Nx;
    int X = (tid - (Z*Nx*Ny)-Y*Nx);
    
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
    float fo0,fo1,fo2,fo3,fo4,fo5,fo6,fo7,fo8,fo9,fo10,fo11,fo12,fo13,fo14;
    float cu,fe_tmp;
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

    float up,vp,wp;


    if(vel_nl[tid]==1){
      up=u_p[tid];vp=v_p[tid];wp=w_p[tid];

      //speed 1 ex=1 ey=ez=0. w=1./9.
      cu=3.*(1.)*(up-ux);
      f1+=(1./9.)*rho*cu;

      //speed 2 ex=-1 ey=ez=0. w=1./9.
      cu=3.*(-1.)*(up-ux);
      f2+=(1./9.)*rho*cu;

      //speed 3 ey=1; ex=ez=0; w=1./9.
      cu=3.*(1.)*(vp-uy);
      f3+=(1./9.)*rho*cu;

      //speed 4 ey=-1; ex=ez=0; w=1./9.
      cu=3.*(-1.)*(vp-uy);
      f4+=(1./9.)*rho*cu;

      //speed 5 ex=ey=0; ez=1; w=1./9.
      cu=3.*(1.)*(wp-uz);
      f5+=(1./9.)*rho*cu;

      //speed 6 ex=ey=0; ez=-1; w=1./9.
      cu=3.*(-1.)*(wp-uz);
      f6+=(1./9.)*rho*cu;

      //speed 7 ex=ey=ez=1; w=1./72.
      cu=3.*((1.)*(up-ux)+(1.)*(vp-uy)+(1.)*(wp-uz));
      f7+=(1./72.)*rho*cu;

      //speed 8 ex=-1 ey=ez=1; w=1./72.
      cu=3.*((-1.)*(up-ux)+(1.)*(vp-uy)+(1.)*(wp-uz));
      f8+=(1./72.)*rho*cu;

      //speed 9 ex=1 ey=-1 ez=1
      cu=3.0*((1.)*(up-ux)+(-1.)*(vp-uy)+(1.)*(wp-uz));
      f9+=(1./72.)*rho*cu;

      //speed 10 ex=-1 ey=-1 ez=1
      cu=3.0*((-1.)*(up-ux)+(-1.)*(vp-uy)+(1.)*(wp-uz));
      f10+=(1./72.)*rho*cu;

      //speed 11 ex=1 ey=1 ez=-1
      cu=3.0*((1.)*(up-ux) +(1.)*(vp-uy)+(-1.)*(wp-uz));
      f11+=(1./72.)*rho*cu;

      //speed 12 ex=-1 ey=1 ez=-1
      cu=3.0*((-1.)*(up-ux)+(1.)*(vp-uy)+(-1.)*(wp-uz));
      f12+=(1./72.)*rho*cu;

      //speed 13 ex=1 ey=-1 ez=-1 w=1./72.
      cu=3.0*((1.)*(up-ux)+(-1.)*(vp-uy)+(-1.)*(wp-uz));
      f13+=(1./72.)*rho*cu;
      
      //speed 14 ex=ey=ez=-1 w=1./72.
      cu=3.0*((-1.)*(up-ux) + (-1.)*(vp-uy) +(-1.)*(wp-uz));
      f14+=(1./72.)*rho*cu;

      ux=up; uy=vp; uz=wp;

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


      //for turbuence modeling, re-use fo# register variables
      //to hold the off-equilibrium value that will be used first
      //to compute a local value of omega using the turbulence model,
      //and ultimately for relaxation

      //relax
      //speed 0 ex=ey=ez=0 w=2./9.
      float fEq;
      fEq=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
      //f0=f0-omega*(f0-fEq);
      fo0 = f0-fEq;

      //speed 1 ex=1 ey=ez=0 w=1./9.
      cu=3.*(1.*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f1=f1-omega*(f1-fEq);
      fo1 = f1-fEq;


      //speed 2 ex=-1 ey=ez=0 w=1./9.
      cu=3.*((-1.)*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f2=f2-omega*(f2-fEq);
      fo2 = f2-fEq;

      //speed 3 ex=0 ey=1 ez=0 w=1./9.
      cu=3.*(1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f3=f3-omega*(f3-fEq);
      fo3=f3-fEq;

      //speed 4 ex=0 ey=-1 ez=0 w=1./9.
      cu=3.*(-1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f4=f4-omega*(f4-fEq);
      fo4=f4-fEq;

      //speed 5 ex=ey=0 ez=1 w=1./9.
      cu=3.*(1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f5=f5-omega*(f5-fEq);
      fo5=f5-fEq;

      //speed 6 ex=ey=0 ez=-1 w=1./9.
      cu=3.*(-1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      //f6=f6-omega*(f6-fEq);
      fo6=f6-fEq;

      //speed 7 ex=ey=ez=1 w=1./72.
      cu=3.*(ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f7=f7-omega*(f7-fEq);
      fo7=f7-fEq;

      //speed 8 ex=-1 ey=ez=1 w=1./72.
      cu=3.*(-ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f8=f8-omega*(f8-fEq);
      fo8=f8-fEq;

      //speed 9 ex=1 ey=-1 ez=1 w=1./72.
      cu=3.*(ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f9=f9-omega*(f9-fEq);
      fo9=f9-fEq;

      //speed 10 ex=-1 ey=-1 ez=1 w=1/72
      cu=3.*(-ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f10=f10-omega*(f10-fEq);
      fo10=f10-fEq;

      //speed 11 ex=1 ey=1 ez=-1 w=1/72
      cu=3.*(ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f11=f11-omega*(f11-fEq);
      fo11=f11-fEq;

      //speed 12 ex=-1 ey=1 ez=-1 w=1/72
      cu=3.*(-ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f12=f12-omega*(f12-fEq);
      fo12=f12-fEq;

      //speed 13 ex=1 ey=ez=-1 w=1/72
      cu=3.*(ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f13=f13-omega*(f13-fEq);
      fo13=f13-fEq;

      //speed 14 ex=ey=ez=-1 w=1/72
      cu=3.*(-ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      //f14=f14-omega*(f14-fEq);
      fo14=f14-fEq;

      //now compute the F-norm of the local stress tensor
      fe_tmp = 0.; cu=0.;
      //1st column = fo1+fo2+fo7+fo8+fo9+...fo14;
      cu = fo1+fo2+fo7+fo8+fo9+fo10+fo11+fo12+fo13+fo14;
      fe_tmp+=cu*cu;
      //2nd column =fo7-fo8-fo9+fo10+fo11-fo12-fo13+fo14
      cu = fo7-fo8-fo9+fo10+fo11-fo12-fo13+fo14;
      fe_tmp+=2.0*cu*cu; //also account for column 4
      //3rd column = fo7-fo8+fo9-fo10-fo11+fo12-fo13+fo14
      cu = fo7-fo8+fo9-fo10-fo11+fo12-fo13+fo14;
      fe_tmp+=2.0*cu*cu; //also account for column 7
      //4th column = fo7-fo8-fo9+fo10+fo11-fo12-fo13+fo14 (same as column 2)

      //5th column = fo3+fo4+fo7+fo8+fo9+fo10+fo11+fo12+fo13+fo14
      cu = fo3+fo4+fo7+fo8+fo9+fo10+fo11+fo12+fo13+fo14;
      fe_tmp+=cu*cu;
      //6th column = fo7+fo8-fo9-fo10-fo11-fo12+fo13+fo14
      cu = fo7+fo8-fo9-fo10-fo11-fo12+fo13+fo14;
      fe_tmp+=2.0*cu*cu; //to account for column 8
      //7th column = fo7-fo8+fo9-fo10-fo11+fo12-fo13+fo14 (same as column 3)

      //8th column = same as colum 6

      //9th column = fo5+fo6+fo7+....+fo14
      cu = fo5+fo6+fo7+fo8+fo9+fo10+fo11+fo12+fo13+fo14;
      fe_tmp+=cu*cu;

      fe_tmp=sqrt(fe_tmp);

      cu=get_omega_turb(nu,fe_tmp);
      //now, relax with this omega (cu)
      f0=f0-cu*fo0;
      f1=f1-cu*fo1;
      f2=f2-cu*fo2;
      f3=f3-cu*fo3;
      f4=f4-cu*fo4;
      f5=f5-cu*fo5;
      f6=f6-cu*fo6;
      f7=f7-cu*fo7;
      f8=f8-cu*fo8;
      f9=f9-cu*fo9;
      f10=f10-cu*fo10;
      f11=f11-cu*fo11;
      f12=f12-cu*fo12;
      f13=f13-cu*fo13;
      f14=f14-cu*fo14;


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

  if(nrhs!=13)
    return err("Usage: channel3D_D3Q15_LBGK_turbulence_nodeList_ts(fOut,fIn,snl,vel_nl,u_p,v_p,w_p,nu,Nx,Ny,Nz,ndList,numList)");


  mxArray * m_fOut = prhs[0];
  mxArray * m_fIn=prhs[1];
  mxArray * m_snl=prhs[2];
  mxArray * m_vel_nl=prhs[3];
  mxArray * m_u_p=prhs[4];
  mxArray * m_v_p=prhs[5];
  mxArray * m_w_p=prhs[6];
  float nu = mxGetScalar(prhs[7]);
  int Nx = mxGetScalar(prhs[8]);
  int Ny = mxGetScalar(prhs[9]);
  int Nz = mxGetScalar(prhs[10]);
  mxArray * m_ndList=prhs[11];
  int numList = mxGetScalar(prhs[12]);

 

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
  

 
  int nnodes=Nx*Ny*Nz;
  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  channel3D_D3Q15_LBGK_turb_NL_ts<<<GRIDS,BLOCKS>>>(fOut_d,fIn_d,snl_d,vel_nl_d,
					       u_p_d,v_p_d,w_p_d,nu,
					       Nx,Ny,Nz,ndList,numList);


  return errNone;

}
