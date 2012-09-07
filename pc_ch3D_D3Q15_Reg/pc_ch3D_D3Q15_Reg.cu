#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"

#define SPDS 15
#define TPB 64


__global__ void pc_D3Q15_Reg(float * fIn, float * fEq, float * rho_d, 
			     float * ux_d, float * uy_d, float * uz_d,
			     const int * vw_nl, const float * vw_uz, 
			     const int * pe_nl, const float rho_out, 
			     const int * snl, const int nnodes){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){

    //load density distribution data
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
    f0 = fIn[tid]; f1=fIn[nnodes+tid];
    f2 = fIn[2*nnodes+tid]; f3 = fIn[3*nnodes+tid];
    f4 = fIn[4*nnodes+tid]; f5 = fIn[5*nnodes+tid];
    f6 = fIn[6*nnodes+tid]; f7 = fIn[7*nnodes+tid];
    f8 = fIn[8*nnodes+tid]; f9=fIn[9*nnodes+tid];
    f10=fIn[10*nnodes+tid]; f11=fIn[11*nnodes+tid];
    f12 = fIn[12*nnodes+tid]; f13=fIn[13*nnodes+tid];
    f14=fIn[14*nnodes+tid];
    float ux,uy,uz,rho;
    rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;
    ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux = ux/rho;
    uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy = uy/rho;
    uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz = uz/rho;

    //detect boundary nodes and compute their macroscopic properties.
    if(snl[tid]==1){
      ux=0;uy=0;uz=0;
      ux_d[tid]=0.; uy_d[tid]=0.; uz_d[tid]=0.;
    }

    if(vw_nl[tid]==1){
      ux=0;uy=0; uz=vw_uz[tid];
      ux_d[tid]=0.; uy_d[tid]=0.; uz_d[tid]=uz;
      //set rho based on uz
      rho = 1./(1.-uz)*(2.*(f6+f11+f12+f13+f14)+(f0+f1+f2+f3+f4));
      rho_d[tid]=rho;//update global array
    }

    if(pe_nl[tid]==1){
      rho=rho_out; ux=0.; uy=0.;
      rho_d[tid]=rho; ux_d[tid]=0.; uy_d[tid]=0.;
      //set uz based on rho...
      uz=-1.+((2.*(f5+f7+f8+f9+f10)+(f0+f1+f2+f3+f4)))/rho_out;
      uz_d[tid]=uz;//update global array

    }

    float cu;
    //if it's a boundary node, I compute fEq for everyone so I can
    //store it and manipulate it for the boundary condition.
    //if it's a solid node or an interior node, I just compute fEq and store
    //it to its global value...

    if((vw_nl[tid]==1) | (pe_nl[tid]==1)){
      float fe0,fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9,fe10,fe11,fe12,fe13,fe14;
    
      float ft1,ft2,ft3,ft4,ft5,ft6,ft7,ft8,ft9,ft10,ft11,ft12,ft13,ft14;
      float w;
      //speed 0 ex=ey=ez=0 w=2./9.
    
      fe0=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
      fEq[tid]=fe0;

      //speed 1 ex=1 ey=ez=0 w=1./9.
      cu=3.*(1.*ux);
      fe1=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      fEq[nnodes+tid]=fe1;

      //speed 2 ex=-1 ey=ez=0 w=1./9.
      cu=3.*((-1.)*ux);
      fe2=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      fEq[2*nnodes+tid]=fe2;

      //speed 3 ex=0 ey=1 ez=0 w=1./9.
      cu=3.*(1.*uy);
      fe3=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      fEq[3*nnodes+tid]=fe3;

      //speed 4 ex=0 ey=-1 ez=0 w=1./9.
      cu=3.*(-1.*uy);
      fe4=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      fEq[4*nnodes+tid]=fe4;

      //speed 5 ex=ey=0 ez=1 w=1./9.
      cu=3.*(1.*uz);
      fe5=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      fEq[5*nnodes+tid]=fe5;

      //speed 6 ex=ey=0 ez=-1 w=1./9.
      cu=3.*(-1.*uz);
      fe6=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      fEq[6*nnodes+tid]=fe6;

      //speed 7 ex=ey=ez=1 w=1./72.
      cu=3.*(ux+uy+uz);
      fe7=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      fEq[7*nnodes+tid]=fe7;

      //speed 8 ex=-1 ey=ez=1 w=1./72.
      cu=3.*(-ux+uy+uz);
      fe8=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      fEq[8*nnodes+tid]=fe8;

      //speed 9 ex=1 ey=-1 ez=1 w=1./72.
      cu=3.*(ux-uy+uz);
      fe9=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      fEq[9*nnodes+tid]=fe9;

      //speed 10 ex=-1 ey=-1 ez=1 w=1/72
      cu=3.*(-ux-uy+uz);
      fe10=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			 1.5*(ux*ux+uy*uy+uz*uz));
      fEq[10*nnodes+tid]=fe10;

      //speed 11 ex=1 ey=1 ez=-1 w=1/72
      cu=3.*(ux+uy-uz);
      fe11=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			 1.5*(ux*ux+uy*uy+uz*uz));
      fEq[11*nnodes+tid]=fe11;

      //speed 12 ex=-1 ey=1 ez=-1 w=1/72
      cu=3.*(-ux+uy-uz);
      fe12=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			 1.5*(ux*ux+uy*uy+uz*uz));
      fEq[12*nnodes+tid]=fe12;

      //speed 13 ex=1 ey=ez=-1 w=1/72
      cu=3.*(ux-uy-uz);
      fe13=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			 1.5*(ux*ux+uy*uy+uz*uz));
      fEq[13*nnodes+tid]=fe13;

      //speed 14 ex=ey=ez=-1 w=1/72
      cu=3.*(-ux-uy-uz);
      fe14=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			 1.5*(ux*ux+uy*uy+uz*uz));
      fEq[14*nnodes+tid]=fe14;

      
      
      if(vw_nl[tid]==1){
	//take actions for the west velocity node list

	//adjust fIn for the unknown velocities: 5,7,8,9,10
	//bounce-back of non-equilibrium parts
	//f5, bb_spd=f6
	f5=fe5+(f6-fe6);
	//f7, bb_spd=f14
	f7=fe7+(f14-fe14);
	//f8, bb_spd=f13
	f8=fe8+(f13-fe13);
	//f9, bb_spd=f12
	f9=fe9+(f12-fe12);
	//f10, bb_spd=f11
	f10=fe10+(f11-fe11);

      }else{
	//take actions for the east pressure node list
	f6=fe6+(f5-fe5);
	f11=f11+(f10-fe10);
	f12=f12+(f9-fe9);
	f13=f13+(f8-fe8);
	f14=f14+(f7-fe7);
      }

      //get the non-equilibrium part of each speed
      //ft0=f0-fe0;
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

      //f#=f#*(9/2)*w#

      //f0, still equals 0..
      cu = 9./2.; w = 1./9.;

      //fIn[..] = fe#+f#
      fIn[tid]=fe0;

      fIn[nnodes+tid]=fe1+f1*(cu)*w;
      fIn[2*nnodes+tid]=fe2+f2*(cu)*w;
      fIn[3*nnodes+tid]=fe3+f3*cu*w;
      fIn[4*nnodes+tid]=fe4+f4*cu*w;
      fIn[5*nnodes+tid]=fe5+f5*cu*w;
      fIn[6*nnodes+tid]=fe6+f6*cu*w;
      w = 1./72.;
      fIn[7*nnodes+tid]=fe7+f7*cu*w;
      fIn[8*nnodes+tid]=fe8+f8*cu*w;
      fIn[9*nnodes+tid]=fe9+f9*cu*w;
      fIn[10*nnodes+tid]=fe10+f10*cu*w;
      fIn[11*nnodes+tid]=fe11+f11*cu*w;
      fIn[12*nnodes+tid]=fe12+f12*cu*w;
      fIn[13*nnodes+tid]=fe13+f13*cu*w;
      fIn[14*nnodes+tid]=fe14+f14*cu*w;

        
    }else{
      //speed 0 ex=ey=ez=0, w= 2./9.
      fEq[tid]=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
      

      //speed 1 ex=1 ey=ez=0 w=1./9.
      cu=3.*(1.*ux);
      fEq[nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
				   1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 2 ex=-1 ey=ez=0 w=1./9.
      cu=3.*((-1.)*ux);
      fEq[2*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
				     1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 3 ex=0 ey=1 ez=0 w=1./9.
      cu=3.*(1.*uy);
      fEq[3*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
				     1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 4 ex=0 ey=-1 ez=0 w=1./9.
      cu=3.*(-1.*uy);
      fEq[4*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
				     1.5*(ux*ux+uy*uy+uz*uz));
      

      //speed 5 ex=ey=0 ez=1 w=1./9.
      cu=3.*(1.*uz);
      fEq[5*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
				     1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 6 ex=ey=0 ez=-1 w=1./9.
      cu=3.*(-1.*uz);
      fEq[6*nnodes+tid]=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
				     1.5*(ux*ux+uy*uy+uz*uz));
      

      //speed 7 ex=ey=ez=1 w=1./72.
      cu=3.*(ux+uy+uz);
      fEq[7*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				      1.5*(ux*ux+uy*uy+uz*uz));
      

      //speed 8 ex=-1 ey=ez=1 w=1./72.
      cu=3.*(-ux+uy+uz);
      fEq[8*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				      1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 9 ex=1 ey=-1 ez=1 w=1./72.
      cu=3.*(ux-uy+uz);
      fEq[9*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				      1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 10 ex=-1 ey=-1 ez=1 w=1/72
      cu=3.*(-ux-uy+uz);
      fEq[10*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				       1.5*(ux*ux+uy*uy+uz*uz));
      

      //speed 11 ex=1 ey=1 ez=-1 w=1/72
      cu=3.*(ux+uy-uz);
      fEq[11*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				       1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 12 ex=-1 ey=1 ez=-1 w=1/72
      cu=3.*(-ux+uy-uz);
      fEq[12*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				       1.5*(ux*ux+uy*uy+uz*uz));
     

      //speed 13 ex=1 ey=ez=-1 w=1/72
      cu=3.*(ux-uy-uz);
      fEq[13*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				       1.5*(ux*ux+uy*uy+uz*uz));
      

      //speed 14 ex=ey=ez=-1 w=1/72
      cu=3.*(-ux-uy-uz);
      fEq[14*nnodes+tid]=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
				       1.5*(ux*ux+uy*uy+uz*uz));
      

    }

    



  }
}


err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray *prhs[]){

  if(nrhs!=12)
    return err("Usage: pc_ch3D_D3Q15_Reg(fIn,fEq,rho,ux,uy,uz,vw_nl,vw_uz,pe_nl,rho_out,snl,nnodes)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_fEq = prhs[1];
  mxArray * m_rho = prhs[2];
  mxArray * m_ux = prhs[3];
  mxArray * m_uy = prhs[4];
  mxArray * m_uz = prhs[5];
  mxArray * m_vw_nl = prhs[6];
  mxArray * m_vw_uz = prhs[7];
  mxArray * m_pe_nl = prhs[8];
  float rho_out = mxGetScalar(prhs[9]);
  mxArray * m_snl = prhs[10];
  int nnodes = mxGetScalar(prhs[11]);

  float * fIn; 
  float * fEq;
  float * rho;
  float * ux;
  float * uy;
  float * uz;
  int * vw_nl;
  float * vw_uz;
  int * pe_nl;
  int * snl;

  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&fEq,m_fEq);
  jkt_mem((void**)&rho,m_rho);
  jkt_mem((void**)&ux,m_ux);
  jkt_mem((void**)&uy,m_uy);
  jkt_mem((void**)&uz,m_uz);
  jkt_mem((void**)&vw_nl,m_vw_nl);
  jkt_mem((void**)&vw_uz,m_vw_uz);
  jkt_mem((void**)&pe_nl,m_pe_nl);
  jkt_mem((void**)&snl,m_snl);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((nnodes+TPB-1)/TPB,1,1);

  pc_D3Q15_Reg<<<GRIDS,BLOCKS>>>(fIn,fEq,rho,ux,uy,uz,vw_nl,vw_uz,pe_nl,
				 rho_out,snl,nnodes);




  return errNone;

}
