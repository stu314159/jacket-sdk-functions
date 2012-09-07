#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 96


//first do this without the turbulence model...then with.

__global__ void channel2D_VW_PE_MRT_ts(double * fOut, double * fIn, int * inl, 
				      int * onl, int * snl, double * ux_p, 
				      double rho_out, double * omega_op,int Nx, int Ny){
 int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes = Nx*Ny;
  if(tid<nnodes){
    double fi1,fi2,fi3,fi4,fi5,fi6,fi7,fi8,fi9;
     double fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9;
    double fo1,fo2,fo3,fo4,fo5,fo6,fo7,fo8,fo9;

    __shared__ double omega[9][9]; 

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

    //load portion of omega into shared memory
    if(threadIdx.x<81){
      int col=threadIdx.x/9;
      int row = threadIdx.x-col*9;

      omega[row][col]=*(omega_op+9*col+row);

    }

    //compute rho
    double rho = fi1+fi2+fi3+fi4+fi5+fi6+fi7+fi8+fi9;

//compute velocity
    double ux = (1/rho)*(fi2+fi6+fi9 - (fi7+fi4+fi8));
    double uy = (1/rho)*(fi6+fi3+fi7 - (fi8+fi5+fi9));

 //if tid is an inlet node, set inlet Macroscopic and microscopic BC
    if(inl[tid]==1){
     
      ux = ux_p[tid];
      uy = 0.0;
      rho = (1./(1-ux))*(fi1+fi3+fi5+2.0*(fi4+fi7+fi8));

      //now set microscopic bc on the inlet
      fi2 = fi4+(2./3.)*rho*ux;
      fi6=fi8+(0.5)*(fi5-fi3)+(0.5)*rho*uy+(1./6.)*rho*ux;
      fi9=fi7+0.5*(fi3-fi5)-0.5*rho*uy+(1./6.)*rho*ux;

    }//if(inlet_node_list[tid]==1)...

    //if tid is an outlet node, set outlet Macroscopic and microscopic BC
    if(onl[tid]==1){
      //macroscopic BC
      rho = rho_out;
      ux = -1. + (1./rho)*(fi1+fi3+fi5+2.0*(fi2+fi6+fi9));
      uy = 0.;
      //microscopic BC
      fi4 = fi2-(2./3.)*rho*ux;
      fi8=fi6+0.5*(fi3-fi5)+0.5*rho*uy-(1./6.)*rho*ux;
      fi7 = fi9+0.5*(fi5-fi3)+0.5*rho*uy-(1./6.)*rho*ux;

    }//if(outlet_node_list[tid]==1)...

    //everybody compute fEq

    //speed 1
    double w = 4./9.;
    double cu = 0.;
   
    fe1 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    

    //speed 2
    w = 1./9.;
    cu = 3.0*ux;
    fe2 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
   

    //speed 3
    cu = 3.0*uy;
    fe3 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    

    //speed 4
    cu = -3.0*ux;
    fe4 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
   

    //speed 5
    cu = -3.0*uy;
    fe5 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    

    //speed 6
    w = 1./36.;
    cu = 3.0*(ux+uy);
    fe6 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    

    //speed 7
    cu = 3.0*(-ux+uy);
    fe7 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    

    //speed 8
    cu = 3.0*(-ux-uy);
    fe8 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    

    //speed 9
    cu= 3.0*(ux-uy);
    fe9 = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    

    //really...I need fe to equal the non-equilibrium part...
    fe1=fi1-fe1;
    fe2=fi2-fe2;
    fe3=fi3-fe3;
    fe4=fi4-fe4;
    fe5=fi5-fe5;
    fe6=fi6-fe6;
    fe7=fi7-fe7;
    fe8=fi8-fe8;
    fe9=fi9-fe9;

    //MRT relaxation
   
    __syncthreads();//make sure omega is loaded...
    
    fo1=fi1-(fe1*omega[0][0]+fe2*omega[1][0]+fe3*omega[2][0]+fe4*omega[3][0]+fe5*omega[4][0]+fe6*omega[5][0]+fe7*omega[6][0]+fe8*omega[7][0]+fe9*omega[8][0]);

    fo2=fi2-(fe1*omega[0][1]+fe2*omega[1][1]+fe3*omega[2][1]+fe4*omega[3][1]+fe5*omega[4][1]+fe6*omega[5][1]+fe7*omega[6][1]+fe8*omega[7][1]+fe9*omega[8][1]);

    fo3=fi3-(fe1*omega[0][2]+fe2*omega[1][2]+fe3*omega[2][2]+fe4*omega[3][2]+fe5*omega[4][2]+fe6*omega[5][2]+fe7*omega[6][2]+fe8*omega[7][2]+fe9*omega[8][2]);

    fo4=fi4-(fe1*omega[0][3]+fe2*omega[1][3]+fe3*omega[2][3]+fe4*omega[3][3]+fe5*omega[4][3]+fe6*omega[5][3]+fe7*omega[6][3]+fe8*omega[7][3]+fe9*omega[8][3]);

    fo5=fi5-(fe1*omega[0][4]+fe2*omega[1][4]+fe3*omega[2][4]+fe4*omega[3][4]+fe5*omega[4][4]+fe6*omega[5][4]+fe7*omega[6][4]+fe8*omega[7][4]+fe9*omega[8][4]);

    fo6=fi6-(fe1*omega[0][5]+fe2*omega[1][5]+fe3*omega[2][5]+fe4*omega[3][5]+fe5*omega[4][5]+fe6*omega[5][5]+fe7*omega[6][5]+fe8*omega[7][5]+fe9*omega[8][5]);

    fo7=fi7-(fe1*omega[0][6]+fe2*omega[1][6]+fe3*omega[2][6]+fe4*omega[3][6]+fe5*omega[4][6]+fe6*omega[5][6]+fe7*omega[6][6]+fe8*omega[7][6]+fe9*omega[8][6]);

    fo8=fi8-(fe1*omega[0][7]+fe2*omega[1][7]+fe3*omega[2][7]+fe4*omega[3][7]+fe5*omega[4][7]+fe6*omega[5][7]+fe7*omega[6][7]+fe8*omega[7][7]+fe9*omega[8][7]);

    fo9=fi9-(fe1*omega[0][8]+fe2*omega[1][8]+fe3*omega[2][8]+fe4*omega[3][8]+fe5*omega[4][8]+fe6*omega[5][8]+fe7*omega[6][8]+fe8*omega[7][8]+fe9*omega[8][8]);
   


    //bounce-back nodes do this instead...
    if(snl[tid]==1){
      fo1=fi1;
      fo2=fi4; fo4=fi2;
      fo3=fi5; fo5=fi3;
      fo6=fi8; fo8=fi6;
      fo7=fi9; fo9=fi7;
      ux = 0.; uy = 0.;

    }//if(solid_node_list[tid]==1)...

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

  }
} 


err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]){


  if(nrhs!=10)
    return err("Usage: channel2D_VW_PE_MRT_DPts(fOut,fIn,inl,onl,snl,ux_p,rho_out,omega_op,Nx,Ny");

  mxArray * m_fOut = prhs[0];
  mxArray * m_fIn = prhs[1];
  mxArray * m_inl = prhs[2];
  mxArray * m_onl = prhs[3];
  mxArray * m_snl = prhs[4];
  mxArray * m_ux_p = prhs[5];
  double rho_out = mxGetScalar(prhs[6]);
  mxArray *  m_omega_op = prhs[7];
  int Nx = mxGetScalar(prhs[8]);
  int Ny = mxGetScalar(prhs[9]);

  int nnodes = Nx*Ny;

  double * fOut; double * fIn; int * inl; int * onl; int * snl;
  double * ux_p;
double * omega_op;
  jkt_mem((void**)&fOut,m_fOut);
  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&inl,m_inl);
  jkt_mem((void**)&onl,m_onl);
  jkt_mem((void**)&snl,m_snl);
  jkt_mem((void**)&ux_p,m_ux_p);
 jkt_mem((void**)&omega_op,m_omega_op);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);
  channel2D_VW_PE_MRT_ts<<<GRIDS,BLOCKS>>>(fOut,fIn,inl,onl,snl,ux_p,rho_out,omega_op,Nx,Ny);

  return errNone;

}
