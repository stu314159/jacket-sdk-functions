#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 96

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



//first do this without the turbulence model...then with.

__global__ void pois2D_LBGK_ts_kernel(float * fOut, float * fIn, int * inl, 
				      int * onl, int * snl, float u_max, 
				      float rho_out, float nu,int Nx, int Ny){
 int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes = Nx*Ny;
  if(tid<nnodes){
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

//compute velocity
    float ux = (1/rho)*(fi2+fi6+fi9 - (fi7+fi4+fi8));
    float uy = (1/rho)*(fi6+fi3+fi7 - (fi8+fi5+fi9));

 //if tid is an inlet node, set inlet Macroscopic and microscopic BC
    if(inl[tid]==1){
     
      ux = u_max;
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

//compute feq and collide...do it one velocity at a time.
    //speed 1
    float w = 4./9.;
    float cu = 0.;
    float omega = 1./(3.*(nu)+0.5);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo1 = fi1-omega*(fi1-fe_tmp);
    fo1 = (fi1-fe_tmp);

    //speed 2
    w = 1./9.;
    cu = 3.0*ux;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo2 = (fi2-fe_tmp);


    //fo2 = fi2-omega*(fi2-fe_tmp);

    //speed 3
    cu = 3.0*uy;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo3 = fi3-omega*(fi3-fe_tmp);
    fo3=(fi3-fe_tmp);

    //speed 4
    cu = -3.0*ux;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo4=(fi4-fe_tmp);
    //fo4=fi4-omega*(fi4-fe_tmp);

    //speed 5
    cu = -3.0*uy;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo5=(fi5-fe_tmp);
    //fo5=fi5-omega*(fi5-fe_tmp);

    //speed 6
    w = 1./36.;
    cu = 3.0*(ux+uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo6=(fi6-fe_tmp);
    //fo6 = fi6-omega*(fi6-fe_tmp);

    //speed 7
    cu = 3.0*(-ux+uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo7=(fi7-fe_tmp);
    // fo7=fi7-omega*(fi7-fe_tmp);

    //speed 8
    cu = 3.0*(-ux-uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo8=(fi8-fe_tmp);
    //fo8=fi8-omega*(fi8-fe_tmp);

    //speed 9
    cu= 3.0*(ux-uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    fo9=(fi9-fe_tmp);
    //fo9=fi9-omega*(fi9-fe_tmp);

    //now, compute the local stress tensor
    //re-use cu and fe_tmp for this purpose
    // fe_tmp to accumulate the F-norm of the local stress tensor
    // cu, to compute each contributing component.
    fe_tmp = 0.; cu=0.;
    cu = fo2+fo4+fo6+fo7+fo8+fo9;
    fe_tmp+=cu*cu;
    cu=fo6-fo7+fo8-fo9;
    fe_tmp+=2.0*cu*cu;
    cu=fo3+fo5+fo6+fo7+fo8+fo9;
    fe_tmp+=cu*cu;
    fe_tmp = sqrt(fe_tmp); //this is now (I think) the F-norm of the local stress tensor.

    omega = get_omega_turb(nu,fe_tmp);

    //now, relax with this omega
    fo1 = fi1-omega*fo1;
    fo2=fi2-omega*fo2;
    fo3=fi3-omega*fo3;
    fo4=fi4-omega*fo4;
    fo5=fi5-omega*fo5;
    fo6=fi6-omega*fo6;
    fo7=fi7-omega*fo7;
    fo8=fi8-omega*fo8;
    fo9=fi9-omega*fo9;






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
    return err("Usage: pois2D_LBGK_turbulence_ts(fOut,fIn,inl,onl,snl,u_max,rho_out,nu,Nx,Ny");

  mxArray * m_fOut = prhs[0];
  mxArray * m_fIn = prhs[1];
  mxArray * m_inl = prhs[2];
  mxArray * m_onl = prhs[3];
  mxArray * m_snl = prhs[4];
  float u_max = mxGetScalar(prhs[5]);
  float rho_out = mxGetScalar(prhs[6]);
  float nu = mxGetScalar(prhs[7]);
  int Nx = mxGetScalar(prhs[8]);
  int Ny = mxGetScalar(prhs[9]);

  int nnodes = Nx*Ny;

  float * fOut; float * fIn; int * inl; int * onl; int * snl;
  jkt_mem((void**)&fOut,m_fOut);
  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&inl,m_inl);
  jkt_mem((void**)&onl,m_onl);
  jkt_mem((void**)&snl,m_snl);



  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);
  pois2D_LBGK_ts_kernel<<<GRIDS,BLOCKS>>>(fOut,fIn,inl,onl,snl,u_max,rho_out,nu,Nx,Ny);

  return errNone;

}
