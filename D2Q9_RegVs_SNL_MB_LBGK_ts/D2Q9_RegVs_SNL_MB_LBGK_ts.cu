#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 96

__global__ void D2Q9_RegVs_SNL_MB_LBGK_ts(float * fOut,const float *fIn,
					  const int * vnl,
					  const int * snl,const int * mbnl,
					  const float * ux_p, 
					  const float * uy_p,
					  const float nu, const int Nx,
					  const int Ny){
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes = Nx*Ny;
  if(tid<nnodes){
    float fi1,fi2,fi3,fi4,fi5,fi6,fi7,fi8,fi9;
    float fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9;
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
    float w, cu;
    //compute velocity
    float ux = (1/rho)*(fi2+fi6+fi9 - (fi7+fi4+fi8));
    float uy = (1/rho)*(fi6+fi3+fi7 - (fi8+fi5+fi9));

    if(mbnl[tid]==1){
      //moving boundary
      float dx=ux_p[tid]-ux;
      float dy = uy_p[tid]-uy;
     
      //speed 2
      w=1./9.;
      cu = 3.*(dx);
      fi2+=w*rho*cu;

      //speed 3
      cu=3.*dy;
      fi3+=w*rho*cu;

      //speed 4
      cu=3.*(-dx);
      fi4+=w*rho*cu;

      //speed 5
      cu = 3.*(-dy);
      fi5+=w*rho*cu;

      //speed 6
      w=1./36.;
      cu=3.*(dx+dy);
      fi6+=w*rho*cu;

      //speed 7
      cu=3.*(-dx+dy);
      fi7+=w*rho*cu;

      //speed 8
      cu=3.*(-dx-dy);
      fi8+=w*rho*cu;

      //speed 9
      cu=3.*(dx-dy);
      fi9+=w*rho*cu;

      //also, update macroscopic velocity...
      ux +=dx;
      uy+=dy;

    }
    if(vnl[tid]==1){
      ux=0.;
      uy=uy_p[tid]; 
      rho = (1./(1.-uy))*(2.0*(fi5+fi8+fi9)+(fi1+fi2+fi4));


    }
    //everybody compute fEq

    //speed 1
    w = 4./9.;
    cu = 0.;
    float omega = 1./(3.*(nu)+0.5);
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
    
    if(vnl[tid]==1){
      fi3=fe3+(fi5-fe5);
      fi6=fe6+(fi8-fe8);
      fi7=fe7+(fi9-fe9);

      //temporarily store the non-equilibrium part in fo#
      fo1=fi1-fe1;
      fo2=fi2-fe2;
      fo3=fi3-fe3;
      fo4=fi4-fe4;
      fo5=fi5-fe5;
      fo6=fi6-fe6;
      fo7=fi7-fe7;
      fo8=fi8-fe8;
      fo9=fi9-fe9;
      //apply *Q*Q' to the non-equilibrium part and store in f#
   
      fi1= -fo2/3. - fo3/3. - fo4/3. - fo5/3. - (2.*fo6)/3. - (2.*fo7)/3. - (2.*fo8)/3. - (2.*fo9)/3.;
      fi2= (2.*fo2)/3. - fo3/3. + (2.*fo4)/3. - fo5/3. + fo6/3. + fo7/3. + fo8/3. + fo9/3.; 
      fi3=(2.*fo3)/3. - fo2/3. - fo4/3. + (2.*fo5)/3. + fo6/3. + fo7/3. + fo8/3. + fo9/3.; 
      fi4=(2.*fo2)/3. - fo3/3. + (2.*fo4)/3. - fo5/3. + fo6/3. + fo7/3. + fo8/3. + fo9/3.; 
      fi5=(2.*fo3)/3. - fo2/3. - fo4/3. + (2.*fo5)/3. + fo6/3. + fo7/3. + fo8/3. + fo9/3.; 
      fi6=(2.*fo2)/3. + (2.*fo3)/3. + (2.*fo4)/3. + (2.*fo5)/3. + (10.*fo6)/3. - (2.*fo7)/3. + (10.*fo8)/3. - (2.*fo9)/3.;
      fi7=(2.*fo2)/3. + (2.*fo3)/3. + (2.*fo4)/3. + (2.*fo5)/3. - (2.*fo6)/3. + (10.*fo7)/3. - (2.*fo8)/3. + (10.*fo9)/3.; 
      fi8=(2.*fo2)/3. + (2.*fo3)/3. + (2.*fo4)/3. + (2.*fo5)/3. + (10.*fo6)/3. - (2.*fo7)/3. + (10.*fo8)/3. - (2.*fo9)/3.; 
      fi9=(2.*fo2)/3. + (2.*fo3)/3. + (2.*fo4)/3. + (2.*fo5)/3. - (2.*fo6)/3. + (10.*fo7)/3. - (2.*fo8)/3. + (10.*fo9)/3.;

      //apply (1./9.)*w(i)
      w = 4./9.;cu = 1./9.;
      fi1= cu*w*fi1; 
      w = 1./9.;
      fi2=cu*w*fi2;
      fi3=cu*w*fi3;
      fi4=cu*w*fi4;
      fi5=cu*w*fi5;
      w=1./36.;
      fi6=cu*w*fi6;
      fi7=cu*w*fi7;
      fi8=cu*w*fi8;
      fi9=cu*w*fi9;

      fi1+=fe1;
      fi2+=fe2;
      fi3+=fe3;
      fi4+=fe4;
      fi5+=fe5;
      fi6+=fe6;
      fi7+=fe7;
      fi8+=fe8;
      fi9+=fe9;

    }
    
    //relax
    //speed 1
  
    
    fo1 = fi1-omega*(fi1-fe1);

    //speed 2
   
    fo2 = fi2-omega*(fi2-fe2);

    //speed 3
   
    fo3 = fi3-omega*(fi3-fe3);

    //speed 4
   
    fo4=fi4-omega*(fi4-fe4);

    //speed 5
    
    fo5=fi5-omega*(fi5-fe5);

    //speed 6
   
    fo6 = fi6-omega*(fi6-fe6);

    //speed 7
  
    fo7=fi7-omega*(fi7-fe7);

    //speed 8
   
    fo8=fi8-omega*(fi8-fe8);

    //speed 9
   
    fo9=fi9-omega*(fi9-fe9);

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
    return err("Usage:D2Q9_RegVs_SNL_MB_LBGK_ts(fOut,fIn,vnl,snl,mbnl,ux_p,uy_p,nu,Nx,Ny");

  mxArray * m_fOut = prhs[0];
  mxArray * m_fIn = prhs[1];
  mxArray * m_vnl = prhs[2];
  mxArray * m_snl = prhs[3];
  mxArray * m_mbnl = prhs[4];
  mxArray * m_ux_p = prhs[5];
  mxArray * m_uy_p = prhs[6];
  float nu = mxGetScalar(prhs[7]);
  int Nx = mxGetScalar(prhs[8]);
  int Ny = mxGetScalar(prhs[9]);

  float * fOut; float * fIn; int * snl; int * mbnl; int * vnl;
  float * ux_p; float * uy_p;

  jkt_mem((void**)&fOut,m_fOut);
  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&vnl,m_vnl);
  jkt_mem((void**)&snl,m_snl);
  jkt_mem((void**)&mbnl,m_mbnl);
  jkt_mem((void**)&ux_p,m_ux_p);
  jkt_mem((void**)&uy_p,m_uy_p);

  int nnodes = Nx*Ny;

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);
  D2Q9_RegVs_SNL_MB_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut,fIn,vnl,snl,mbnl,ux_p,uy_p,
					      nu,Nx,Ny);

  return errNone;

}
