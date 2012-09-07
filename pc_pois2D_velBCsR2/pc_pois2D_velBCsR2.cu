#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define TPB 128

__global__ void pc_pois2D(float * fEq,float * fIn, float * ux_d, float * uy_d,
			  float * rho_d, int * vnl, int * snl,
			  float * ux_p,float * uy_p, int nnodes){

  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    float fi1,fi2,fi3,fi4,fi5,fi6,fi7,fi8,fi9;
    float fe_tmp,cu,w;
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
    rho_d[tid]=rho;

    //compute velocity
    float ux = (1./rho)*(fi2+fi6+fi9 - (fi7+fi4+fi8));
    float uy = (1./rho)*(fi6+fi3+fi7 - (fi8+fi5+fi9));

    ux_d[tid]=ux; uy_d[tid]=uy;

    if(vnl[tid]==1){
      float dx= ux_p[tid]-ux;
      float dy =uy_p[tid]-uy;
      ux_d[tid]=ux_p[tid];
      uy_d[tid]=uy_p[tid];

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
     
      //write-back fIn
      //fIn[tid]=fi1;
      fIn[tid+nnodes]=fi2;
      fIn[tid+2*nnodes]=fi3;
      fIn[tid+3*nnodes]=fi4;
      fIn[tid+4*nnodes]=fi5;
      fIn[tid+5*nnodes]=fi6;
      fIn[tid+6*nnodes]=fi7;
      fIn[tid+7*nnodes]=fi8;
      fIn[tid+8*nnodes]=fi9;

    }

 if(snl[tid]==0){
    w = 4./9.;
    cu = 0.;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo1 = fi1-omega*(fi1-fe_tmp);
    fEq[tid]=fe_tmp;
    //speed 2
    w = 1./9.;
    cu = 3.0*ux;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo2 = fi2-omega*(fi2-fe_tmp);
    fEq[tid+nnodes]=fe_tmp;

    //speed 3
    cu = 3.0*uy;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo3 = fi3-omega*(fi3-fe_tmp);
    fEq[tid+2*nnodes]=fe_tmp;

    //speed 4
    cu = -3.0*ux;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo4=fi4-omega*(fi4-fe_tmp);
    fEq[tid+3*nnodes]=fe_tmp;

    //speed 5
    cu = -3.0*uy;
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo5=fi5-omega*(fi5-fe_tmp);
    fEq[tid+4*nnodes]=fe_tmp;

    //speed 6
    w = 1./36.;
    cu = 3.0*(ux+uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo6 = fi6-omega*(fi6-fe_tmp);
    fEq[tid+5*nnodes]=fe_tmp;

    //speed 7
    cu = 3.0*(-ux+uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo7=fi7-omega*(fi7-fe_tmp);
    fEq[tid+6*nnodes]=fe_tmp;

    //speed 8
    cu = 3.0*(-ux-uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo8=fi8-omega*(fi8-fe_tmp);
    fEq[tid+7*nnodes]=fe_tmp;

    //speed 9
    cu= 3.0*(ux-uy);
    fe_tmp = w*rho*(1.+cu+0.5*(cu*cu)-1.5*(ux*ux + uy*uy));
    //fo9=fi9-omega*(fi9-fe_tmp);
    fEq[tid+8*nnodes]=fe_tmp;

    }
    if(snl[tid]==1){
      ux_d[tid]=0.; uy_d[tid]=0.;
    }
   

  }
}



err_t jktFunction(int nlhs,mxArray * plhs[], int nrhs, mxArray * prhs[]){

  if(nrhs!=10)
    return err("Usage: pc_pois2D_velBCsR2(fEq,fIn,ux,uy,rho,vnl,snl,ux_p,uy_p,nnodes)");

  mxArray * m_fEq=prhs[0];
  mxArray * m_fIn=prhs[1];
  mxArray * m_ux=prhs[2];
  mxArray * m_uy=prhs[3];
  mxArray * m_rho=prhs[4];
  mxArray * m_vnl=prhs[5];
  
  mxArray * m_snl=prhs[6];
  mxArray * m_ux_p=prhs[7];
  mxArray * m_uy_p=prhs[8];
  int nnodes = mxGetScalar(prhs[9]);
 
  // const mwSize * dims;
  // int stat=jkt_dims(m_fIn,&dims);
  // int nnodes=dims[0];

  float * fEq_d;
  float * fIn_d;
  float * ux_d;
  float * uy_d;
  float * rho_d;
  int * vnl_d;
  int * snl_d;
  float * ux_p_d;
  float * uy_p_d;

  jkt_mem((void**)&fEq_d,m_fEq);
  jkt_mem((void**)&fIn_d,m_fIn);
  jkt_mem((void**)&ux_d,m_ux);
  jkt_mem((void**)&uy_d,m_uy);
  jkt_mem((void**)&rho_d,m_rho);
  jkt_mem((void**)&vnl_d,m_vnl);
 
  jkt_mem((void**)&snl_d,m_snl);
  jkt_mem((void**)&ux_p_d,m_ux_p);
  jkt_mem((void**)&uy_p_d,m_uy_p);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  pc_pois2D<<<GRIDS,BLOCKS>>>(fEq_d,fIn_d,ux_d,uy_d,rho_d,
			      vnl_d,snl_d,ux_p_d,uy_p_d,nnodes);

  return errNone;

}