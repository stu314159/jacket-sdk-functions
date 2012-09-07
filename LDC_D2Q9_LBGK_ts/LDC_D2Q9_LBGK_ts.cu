#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 16

__global__ void ldc_D2Q9_LBGK_ts(float * fOut, float * fIn, float * U,
				 float * rho_p, const float u_bc, 
				 const float omega,
				 const int Nx, const int Ny){
  int X=threadIdx.x+blockIdx.x*blockDim.x;
  int Y=threadIdx.y+blockIdx.y*blockDim.y;
  if((X<Nx)&&(Y<Ny)){
    int tid=X+Y*Nx;
    int nnodes=Nx*Ny;
    float fi1,fi2,fi3,fi4,fi5,fi6,fi7,fi8,fi9;
    float fe_tmp;
    float fo1,fo2,fo3,fo4,fo5,fo6,fo7,fo8,fo9;

    //get the density data for the lattice point.
    fi1=fIn[tid];
    fi2=fIn[nnodes+tid];
    fi3=fIn[2*nnodes+tid];
    fi4=fIn[3*nnodes+tid];
    fi5=fIn[4*nnodes+tid];
    fi6=fIn[5*nnodes+tid];
    fi7=fIn[6*nnodes+tid];
    fi8=fIn[7*nnodes+tid];
    fi9=fIn[8*nnodes+tid];

    //compute rho
    float rho = fi1+fi2+fi3+fi4+fi5+fi6+fi7+fi8+fi9;
    rho_p[tid]=rho;

    //compute velocity
    float ux = (1/rho)*(fi2+fi6+fi9 - (fi7+fi4+fi8));
    float uy = (1/rho)*(fi6+fi3+fi7 - (fi8+fi5+fi9));
    U[tid]=sqrt(ux*ux+uy*uy);

    if((Y==(Ny-1))&&(!((X==0)||(X==(Nx-1))))){//lid node...
      ux = u_bc;
      uy = 0.0;
      rho = (1./(1.+uy))*
    	((fi1+fi2+fi4)+2.0*(fi3+fi6+fi7));
      fi5 = fi3-(2./3.)*rho*uy;
      fi9 = fi7+0.5*(fi4-fi2)+
    	0.5*rho*ux - (1./6.)*rho*uy;
      fi8=fi6+0.5*(fi2-fi4)-0.5*rho*ux -
    	(1./6.)*rho*uy;

    }
    
    //compute feq and collide...do it one velocity at a time.
    //speed 1
    float w = 4./9.;
    float cu = 0.;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo1 = fi1-omega*(fi1-fe_tmp);

    //speed 2
    w = 1./9.;
    cu = 3.0*ux;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo2 = fi2-omega*(fi2-fe_tmp);

    //speed 3
    cu = 3.0*uy;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo3 = fi3-omega*(fi3-fe_tmp);

    //speed 4
    cu = -3.0*ux;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo4=fi4-omega*(fi4-fe_tmp);

    //speed 5
    cu = -3.0*uy;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo5=fi5-omega*(fi5-fe_tmp);

    //speed 6
    w = 1./36.;
    cu = 3.0*(ux+uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo6 = fi6-omega*(fi6-fe_tmp);

    //speed 7
    cu = 3.0*(-ux+uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo7=fi7-omega*(fi7-fe_tmp);

    //speed 8
    cu = 3.0*(-ux-uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo8=fi8-omega*(fi8-fe_tmp);

    //speed 9
    cu= 3.0*(ux-uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo9=fi9-omega*(fi9-fe_tmp);


    if(((X==0)||(X==(Nx-1))||(Y==0))){//solid node

      fo1=fi1;
      fo2=fi4; fo4=fi2;
      fo3=fi5; fo5=fi3;
      fo6=fi8; fo8=fi6;
      fo7=fi9; fo9=fi7;
      ux = 0.; uy = 0.;
    }


    // stream the result...
    //compute the local stream vector...	
    int x;
    int y;
    int yn;
    int ys;
    int xe;
    int xw;
   
    //int dir; 
    int dof_num; //int f_num;
    x = tid%Nx+1;
    y = ((tid+1)-x+1)/Nx + 1; 

    yn = y%Ny+1;
    xe = x%Nx+1;

    if(y==1){
      ys = Ny;
    }else{
      ys = y-1;
    }
    if(x==1){
      xw=Nx;
    }else{
      xw=x-1;
    }

    dof_num = Nx*(y-1)+x;
    fOut[dof_num-1]=fo1;

    dof_num=Nx*(y-1)+xe;
    fOut[nnodes+dof_num-1]=fo2;

    dof_num=Nx*(yn-1)+x;
    fOut[2*nnodes+dof_num-1]=fo3;

    dof_num=Nx*(y-1)+xw;
    fOut[3*nnodes+dof_num-1]=fo4;

    dof_num=Nx*(ys-1)+x;
    fOut[4*nnodes+dof_num-1]=fo5;

    dof_num=Nx*(yn-1)+xe;
    fOut[5*nnodes+dof_num-1]=fo6;

    dof_num=Nx*(yn-1)+xw;
    fOut[6*nnodes+dof_num-1]=fo7;

    dof_num=Nx*(ys-1)+xw;
    fOut[7*nnodes+dof_num-1]=fo8;

    dof_num=Nx*(ys-1)+xe;
    fOut[8*nnodes+dof_num-1]=fo9;


  }//if(X<Nx...
}

err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs,mxArray * prhs[]){


  mxArray * m_fOut=prhs[0];
  mxArray * m_fIn=prhs[1];
  mxArray * m_U = prhs[2];
  mxArray * m_rho =prhs[3];
  float u_bc=mxGetScalar(prhs[4]);
  float omega=mxGetScalar(prhs[5]);
  int Nx = mxGetScalar(prhs[6]);
  int Ny =mxGetScalar(prhs[7]);

  float * fOut_d;
  float * fIn_d;
  float * U_d;
  float * rho_d;

  jkt_mem((void**)&fOut_d,m_fOut);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&U_d,m_U);
  jkt_mem((void**)&rho_d,m_rho);

  dim3 BLOCKS(TPB,TPB,1);
  dim3 GRIDS((Nx+TPB-1)/TPB,(Ny+TPB-1)/TPB,1);

  ldc_D2Q9_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut_d,fIn_d,U_d,rho_d,u_bc,omega,Nx,Ny);

  return errNone;

}
