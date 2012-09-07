#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 64



__global__ void D3Q19_RegVwPe_SNL_VNL_MRT_NodeList_ts(const float * fIn, float * fOut,
						      const int * SNL,
						      const int * VW_nl, 
						      const int * PE_nl, const float * rho_out,
						      const int * VNL, const float * ux_p,
						      const float * uy_p, const float * uz_p,
						      const float * M,
						      const int Nx, const int Ny, const int Nz,
						      const int * NodeList, const int numNL)
{

  int tid_l=threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes=Nx*Ny*Nz;
  if(tid_l<numNL){
    int tid = NodeList[tid_l]-1;//<--make one based...
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18;
    float cu;
    float w;

    __shared__ float omega[19][19];

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

    //load omega into shared memory.  Remember omega is in row-major order...
    for(int om_bit = threadIdx.x;om_bit<(19*19);om_bit+=blockDim.x){
      int col = om_bit/19;
      int row = om_bit - col*19;
      omega[row][col]=*(M+om_bit);

    }

    //compute density and velocity
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f15-f16+f17-f18; uy/=rho;
    float uz=f5-f6+f11+f12-f13-f14+f15+f16-f17-f18; uz/=rho;

    //take appropriate action if on PE_nl or VW_nl
    if(VW_nl[tid]==1){
      ux=0.;uy=0.; uz=uz_p[tid];
      //set rho based on uz
      rho = (1./(1.-uz))*(2.0*(f6+f13+f14+f17+f18)+(f0+f1+f2+f3+f4+f7+f8+f9+f10));

    }
    if(PE_nl[tid]==1){
      ux=0.; uy=0.; rho=rho_out[tid];
      uz = -1.+((2.*(f5+f11+f12+f15+f16)+(f0+f1+f2+f3+f4+f7+f8+f9+f10)))/rho;

    }
    if(SNL[tid]==1){
      ux=0.; uy=0.; uz=0.;
    }
    if(VNL[tid]==1){

      float dx, dy, dz;
      dx = ux_p[tid]-ux;
      dy = uy_p[tid]-uy;
      dz = uz_p[tid]-uz;

      //speed 1 (1,0,0) w=1./18.

      w = 1./18.;
      cu = 3.*(dx);
      f1+=w*rho*cu;

      //speed 2 (-1,0,0) 
      cu=3.*(-dx);
      f2+=w*rho*cu;

      //speed 3 (0,1,0)
      cu = 3.*(dy);
      f3+=w*rho*cu;

      //speed 4 (0,-1,0)
      cu = 3.*(-dy);
      f4+=w*rho*cu;

      //speed 5 (0,0,1)
      cu = 3.*(dz);
      f5+=w*rho*cu;

      //speed 6 (0,0,-1)
      cu = 3.*(-dz);
      f6+=w*rho*cu;

      w = 1./36.;
      //speed 7 (1,1,0)
      cu = 3.*((dx)+(dy));
      f7+=w*rho*cu;

      //speed 8 ( -1,1,0)
      cu = 3.*((-dx) + (dy));
      f8+=w*rho*cu;

      //speed 9 (1,-1,0)
      cu = 3.*((dx) -(dy));
      f9+=w*rho*cu;

      //speed 10 (-1,-1,0)
      cu = 3.*(-(dx) -(dy));
      f10+=w*rho*cu;

      //speed 11 (1,0,1)
      cu = 3.*((dx)+(dz));
      f11+=w*rho*cu;

      //speed 12 (-1,0,1)
      cu = 3.*(-dx +dz);
      f12+=w*rho*cu;

      //speed 13 (1,0,-1)
      cu = 3.*(dx - dz );
      f13+=w*rho*cu;

      //speed 14 (-1,0,-1)
      cu = 3.*(-dx-dz);
      f14+=w*rho*cu;

      //speed 15 ( 0,1,1)
      cu = 3.*((dy)+dz);
      f15+=w*rho*cu;

      //speed 16 (0,-1,1)
      cu = 3.*(-(dy)+dz);
      f16+=w*rho*cu;

      //speed 17 (0,1,-1)
      cu = 3.*((dy)-dz);
      f17+=w*rho*cu;

      //speed 18 (0,-1,-1)
      cu = 3.*(-dy-dz);
      f18+=w*rho*cu;

      ux+=dx; uy+=dy; uz+=dz;


    }

    //everyone compute equilibrium
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


    if((VW_nl[tid]==1) || (PE_nl[tid]==1)){
      //float ft0;
      float ft1,ft2,ft3,ft4,ft5,ft6,ft7,ft8,ft9,ft10,ft11,ft12,ft13,ft14,ft15,ft16,ft17,ft18;
      if(VW_nl[tid]==1){
	//bounce-back of non-equilibrium for unknown velocities on west boundary
	f5=fe5+(f6-fe6); 
	f11=fe11+(f14-fe14);
	f12=fe12+(f13-fe13);
	f15=fe15+(f18-fe18);
	f16=fe16+(f17-fe17);

      }else{
	//bounce-back of non-equilibrium on east boundary
	f6=fe6+(f5-fe5);
	f13=fe13+(f12-fe12);
	f14=fe14+(f11-fe11);
	f17=fe17+(f16-fe16);
	f18=fe18+(f15-fe15);

      }

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
      ft15=f15-fe15;
      ft16=f16-fe16;
      ft17=f17-fe17;
      ft18=f18-fe18;

      // //apply the tensors...
     

      f0= - ft1/3. - ft2/3. - ft3/3. - ft4/3. - ft5/3. - ft6/3. - (2.*ft7)/3. - (2.*ft8)/3. - (2.*ft9)/3. - (2.*ft10)/3. - (2.*ft11)/3. - (2.*ft12)/3. - (2.*ft13)/3. - (2.*ft14)/3. - (2.*ft15)/3. - (2.*ft16)/3. - (2.*ft17)/3. - (2.*ft18)/3.; 
      f1=(2.*ft1)/3. + (2.*ft2)/3. - ft3/3. - ft4/3. - ft5/3. - ft6/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. - (2.*ft15)/3. - (2.*ft16)/3. - (2.*ft17)/3. - (2.*ft18)/3.; 
      f2=(2.*ft1)/3. + (2.*ft2)/3. - ft3/3. - ft4/3. - ft5/3. - ft6/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. - (2.*ft15)/3. - (2.*ft16)/3. - (2.*ft17)/3. - (2.*ft18)/3.; 
      f3=(2.*ft3)/3. - ft2/3. - ft1/3. + (2.*ft4)/3. - ft5/3. - ft6/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. - (2.*ft11)/3. - (2.*ft12)/3. - (2.*ft13)/3. - (2.*ft14)/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f4=(2.*ft3)/3. - ft2/3. - ft1/3. + (2.*ft4)/3. - ft5/3. - ft6/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. - (2.*ft11)/3. - (2.*ft12)/3. - (2.*ft13)/3. - (2.*ft14)/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f5=(2.*ft5)/3. - ft2/3. - ft3/3. - ft4/3. - ft1/3. + (2.*ft6)/3. - (2.*ft7)/3. - (2.*ft8)/3. - (2.*ft9)/3. - (2.*ft10)/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f6=(2.*ft5)/3. - ft2/3. - ft3/3. - ft4/3. - ft1/3. + (2.*ft6)/3. - (2.*ft7)/3. - (2.*ft8)/3. - (2.*ft9)/3. - (2.*ft10)/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f7=(2.*ft1)/3. + (2.*ft2)/3. + (2.*ft3)/3. + (2.*ft4)/3. - ft5/3. - ft6/3. + (10.*ft7)/3. - (2.*ft8)/3. - (2.*ft9)/3. + (10.*ft10)/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f8=(2.*ft1)/3. + (2.*ft2)/3. + (2.*ft3)/3. + (2.*ft4)/3. - ft5/3. - ft6/3. - (2.*ft7)/3. + (10.*ft8)/3. + (10.*ft9)/3. - (2.*ft10)/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f9=(2.*ft1)/3. + (2.*ft2)/3. + (2.*ft3)/3. + (2.*ft4)/3. - ft5/3. - ft6/3. - (2.*ft7)/3. + (10.*ft8)/3. + (10.*ft9)/3. - (2.*ft10)/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f10=(2.*ft1)/3. + (2.*ft2)/3. + (2.*ft3)/3. + (2.*ft4)/3. - ft5/3. - ft6/3. + (10.*ft7)/3. - (2.*ft8)/3. - (2.*ft9)/3. + (10.*ft10)/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f11=(2.*ft1)/3. + (2.*ft2)/3. - ft3/3. - ft4/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + (10.*ft11)/3. - (2.*ft12)/3. - (2.*ft13)/3. + (10.*ft14)/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f12=(2.*ft1)/3. + (2.*ft2)/3. - ft3/3. - ft4/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. - (2.*ft11)/3. + (10.*ft12)/3. + (10.*ft13)/3. - (2.*ft14)/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f13=(2.*ft1)/3. + (2.*ft2)/3. - ft3/3. - ft4/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. - (2.*ft11)/3. + (10.*ft12)/3. + (10.*ft13)/3. - (2.*ft14)/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f14=(2.*ft1)/3. + (2.*ft2)/3. - ft3/3. - ft4/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + (10.*ft11)/3. - (2.*ft12)/3. - (2.*ft13)/3. + (10.*ft14)/3. + ft15/3. + ft16/3. + ft17/3. + ft18/3.; 
      f15=(2.*ft3)/3. - ft2/3. - ft1/3. + (2.*ft4)/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + (10.*ft15)/3. - (2.*ft16)/3. - (2.*ft17)/3. + (10.*ft18)/3.; 
      f16=(2.*ft3)/3. - ft2/3. - ft1/3. + (2.*ft4)/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. - (2.*ft15)/3. + (10.*ft16)/3. + (10.*ft17)/3. - (2.*ft18)/3.; 
      f17=(2.*ft3)/3. - ft2/3. - ft1/3. + (2.*ft4)/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. - (2.*ft15)/3. + (10.*ft16)/3. + (10.*ft17)/3. - (2.*ft18)/3.; 
      f18=(2.*ft3)/3. - ft2/3. - ft1/3. + (2.*ft4)/3. + (2.*ft5)/3. + (2.*ft6)/3. + ft7/3. + ft8/3. + ft9/3. + ft10/3. + ft11/3. + ft12/3. + ft13/3. + ft14/3. + (10.*ft15)/3. - (2.*ft16)/3. - (2.*ft17)/3. + (10.*ft18)/3.;

      //update fIn for all velocities based on this result.
      cu= 9./2.; w = 1./3.;
      f0=fe0+f0*cu*w;
      w=1./18.;
      f1=fe1+f1*cu*w;
      f2=fe2+f2*cu*w;
      f3=fe3+f3*cu*w;
      f4=fe4+f4*cu*w;
      f5=fe5+f5*cu*w;
      f6=fe6+f6*cu*w;
      w=1./36.;
      f7=fe7+f7*cu*w;
      f8=fe8+f8*cu*w;
      f9=fe9+f9*cu*w;
      f10=fe10+f10*cu*w;
      f11=fe11+f11*cu*w;
      f12=fe12+f12*cu*w;
      f13=fe13+f13*cu*w;
      f14=fe14+f14*cu*w;
      f15=fe15+f15*cu*w;
      f16=fe16+f16*cu*w;
      f17=fe17+f17*cu*w;
      f18=fe18+f18*cu*w;


    }


    if(SNL[tid]==1){

      //bounce back
      //f0=f0-omega*(f0-fe0);
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
    __syncthreads();
    //collect into non-equilibrium parts
    fe0=f0-fe0;
    fe1=f1-fe1;
    fe2=f2-fe2;
    fe3=f3-fe3;
    fe4=f4-fe4;
    fe5=f5-fe5;
    fe6=f6-fe6;
    fe7=f7-fe7;
    fe8=f8-fe8;
    fe9=f9-fe9;
    fe10=f10-fe10;
    fe11=f11-fe11;
    fe12=f12-fe12;
    fe13=f13-fe13;
    fe14=f14-fe14;
    fe15=f15-fe15;
    fe16=f16-fe16;
    fe17=f17-fe17;
    fe18=f18-fe18;

    f0=f0-(fe0*omega[0][0]+fe1*omega[1][0]+fe2*omega[2][0]+fe3*omega[3][0]+fe4*omega[4][0]+fe5*omega[5][0]+fe6*omega[6][0]+fe7*omega[7][0]+fe8*omega[8][0]+fe9*omega[9][0]+fe10*omega[10][0]+fe11*omega[11][0]+fe12*omega[12][0]+fe13*omega[13][0]+fe14*omega[14][0]+fe15*omega[15][0]+fe16*omega[16][0]+fe17*omega[17][0]+fe18*omega[18][0]);

    f1=f1-(fe0*omega[0][1]+fe1*omega[1][1]+fe2*omega[2][1]+fe3*omega[3][1]+fe4*omega[4][1]+fe5*omega[5][1]+fe6*omega[6][1]+fe7*omega[7][1]+fe8*omega[8][1]+fe9*omega[9][1]+fe10*omega[10][1]+fe11*omega[11][1]+fe12*omega[12][1]+fe13*omega[13][1]+fe14*omega[14][1]+fe15*omega[15][1]+fe16*omega[16][1]+fe17*omega[17][1]+fe18*omega[18][1]);

    f2=f2-(fe0*omega[0][2]+fe1*omega[1][2]+fe2*omega[2][2]+fe3*omega[3][2]+fe4*omega[4][2]+fe5*omega[5][2]+fe6*omega[6][2]+fe7*omega[7][2]+fe8*omega[8][2]+fe9*omega[9][2]+fe10*omega[10][2]+fe11*omega[11][2]+fe12*omega[12][2]+fe13*omega[13][2]+fe14*omega[14][2]+fe15*omega[15][2]+fe16*omega[16][2]+fe17*omega[17][2]+fe18*omega[18][2]);

    f3=f3-(fe0*omega[0][3]+fe1*omega[1][3]+fe2*omega[2][3]+fe3*omega[3][3]+fe4*omega[4][3]+fe5*omega[5][3]+fe6*omega[6][3]+fe7*omega[7][3]+fe8*omega[8][3]+fe9*omega[9][3]+fe10*omega[10][3]+fe11*omega[11][3]+fe12*omega[12][3]+fe13*omega[13][3]+fe14*omega[14][3]+fe15*omega[15][3]+fe16*omega[16][3]+fe17*omega[17][3]+fe18*omega[18][3]);

    f4=f4-(fe0*omega[0][4]+fe1*omega[1][4]+fe2*omega[2][4]+fe3*omega[3][4]+fe4*omega[4][4]+fe5*omega[5][4]+fe6*omega[6][4]+fe7*omega[7][4]+fe8*omega[8][4]+fe9*omega[9][4]+fe10*omega[10][4]+fe11*omega[11][4]+fe12*omega[12][4]+fe13*omega[13][4]+fe14*omega[14][4]+fe15*omega[15][4]+fe16*omega[16][4]+fe17*omega[17][4]+fe18*omega[18][4]);

    f5=f5-(fe0*omega[0][5]+fe1*omega[1][5]+fe2*omega[2][5]+fe3*omega[3][5]+fe4*omega[4][5]+fe5*omega[5][5]+fe6*omega[6][5]+fe7*omega[7][5]+fe8*omega[8][5]+fe9*omega[9][5]+fe10*omega[10][5]+fe11*omega[11][5]+fe12*omega[12][5]+fe13*omega[13][5]+fe14*omega[14][5]+fe15*omega[15][5]+fe16*omega[16][5]+fe17*omega[17][5]+fe18*omega[18][5]);

    f6=f6-(fe0*omega[0][6]+fe1*omega[1][6]+fe2*omega[2][6]+fe3*omega[3][6]+fe4*omega[4][6]+fe5*omega[5][6]+fe6*omega[6][6]+fe7*omega[7][6]+fe8*omega[8][6]+fe9*omega[9][6]+fe10*omega[10][6]+fe11*omega[11][6]+fe12*omega[12][6]+fe13*omega[13][6]+fe14*omega[14][6]+fe15*omega[15][6]+fe16*omega[16][6]+fe17*omega[17][6]+fe18*omega[18][6]);

    f7=f7-(fe0*omega[0][7]+fe1*omega[1][7]+fe2*omega[2][7]+fe3*omega[3][7]+fe4*omega[4][7]+fe5*omega[5][7]+fe6*omega[6][7]+fe7*omega[7][7]+fe8*omega[8][7]+fe9*omega[9][7]+fe10*omega[10][7]+fe11*omega[11][7]+fe12*omega[12][7]+fe13*omega[13][7]+fe14*omega[14][7]+fe15*omega[15][7]+fe16*omega[16][7]+fe17*omega[17][7]+fe18*omega[18][7]);

    f8=f8-(fe0*omega[0][8]+fe1*omega[1][8]+fe2*omega[2][8]+fe3*omega[3][8]+fe4*omega[4][8]+fe5*omega[5][8]+fe6*omega[6][8]+fe7*omega[7][8]+fe8*omega[8][8]+fe9*omega[9][8]+fe10*omega[10][8]+fe11*omega[11][8]+fe12*omega[12][8]+fe13*omega[13][8]+fe14*omega[14][8]+fe15*omega[15][8]+fe16*omega[16][8]+fe17*omega[17][8]+fe18*omega[18][8]);

    f9=f9-(fe0*omega[0][9]+fe1*omega[1][9]+fe2*omega[2][9]+fe3*omega[3][9]+fe4*omega[4][9]+fe5*omega[5][9]+fe6*omega[6][9]+fe7*omega[7][9]+fe8*omega[8][9]+fe9*omega[9][9]+fe10*omega[10][9]+fe11*omega[11][9]+fe12*omega[12][9]+fe13*omega[13][9]+fe14*omega[14][9]+fe15*omega[15][9]+fe16*omega[16][9]+fe17*omega[17][9]+fe18*omega[18][9]);

    f10=f10-(fe0*omega[0][10]+fe1*omega[1][10]+fe2*omega[2][10]+fe3*omega[3][10]+fe4*omega[4][10]+fe5*omega[5][10]+fe6*omega[6][10]+fe7*omega[7][10]+fe8*omega[8][10]+fe9*omega[9][10]+fe10*omega[10][10]+fe11*omega[11][10]+fe12*omega[12][10]+fe13*omega[13][10]+fe14*omega[14][10]+fe15*omega[15][10]+fe16*omega[16][10]+fe17*omega[17][10]+fe18*omega[18][10]);

    f11=f11-(fe0*omega[0][11]+fe1*omega[1][11]+fe2*omega[2][11]+fe3*omega[3][11]+fe4*omega[4][11]+fe5*omega[5][11]+fe6*omega[6][11]+fe7*omega[7][11]+fe8*omega[8][11]+fe9*omega[9][11]+fe10*omega[10][11]+fe11*omega[11][11]+fe12*omega[12][11]+fe13*omega[13][11]+fe14*omega[14][11]+fe15*omega[15][11]+fe16*omega[16][11]+fe17*omega[17][11]+fe18*omega[18][11]);

    f12=f12-(fe0*omega[0][12]+fe1*omega[1][12]+fe2*omega[2][12]+fe3*omega[3][12]+fe4*omega[4][12]+fe5*omega[5][12]+fe6*omega[6][12]+fe7*omega[7][12]+fe8*omega[8][12]+fe9*omega[9][12]+fe10*omega[10][12]+fe11*omega[11][12]+fe12*omega[12][12]+fe13*omega[13][12]+fe14*omega[14][12]+fe15*omega[15][12]+fe16*omega[16][12]+fe17*omega[17][12]+fe18*omega[18][12]);

    f13=f13-(fe0*omega[0][13]+fe1*omega[1][13]+fe2*omega[2][13]+fe3*omega[3][13]+fe4*omega[4][13]+fe5*omega[5][13]+fe6*omega[6][13]+fe7*omega[7][13]+fe8*omega[8][13]+fe9*omega[9][13]+fe10*omega[10][13]+fe11*omega[11][13]+fe12*omega[12][13]+fe13*omega[13][13]+fe14*omega[14][13]+fe15*omega[15][13]+fe16*omega[16][13]+fe17*omega[17][13]+fe18*omega[18][13]);

    f14=f14-(fe0*omega[0][14]+fe1*omega[1][14]+fe2*omega[2][14]+fe3*omega[3][14]+fe4*omega[4][14]+fe5*omega[5][14]+fe6*omega[6][14]+fe7*omega[7][14]+fe8*omega[8][14]+fe9*omega[9][14]+fe10*omega[10][14]+fe11*omega[11][14]+fe12*omega[12][14]+fe13*omega[13][14]+fe14*omega[14][14]+fe15*omega[15][14]+fe16*omega[16][14]+fe17*omega[17][14]+fe18*omega[18][14]);

    f15=f15-(fe0*omega[0][15]+fe1*omega[1][15]+fe2*omega[2][15]+fe3*omega[3][15]+fe4*omega[4][15]+fe5*omega[5][15]+fe6*omega[6][15]+fe7*omega[7][15]+fe8*omega[8][15]+fe9*omega[9][15]+fe10*omega[10][15]+fe11*omega[11][15]+fe12*omega[12][15]+fe13*omega[13][15]+fe14*omega[14][15]+fe15*omega[15][15]+fe16*omega[16][15]+fe17*omega[17][15]+fe18*omega[18][15]);

    f16=f16-(fe0*omega[0][16]+fe1*omega[1][16]+fe2*omega[2][16]+fe3*omega[3][16]+fe4*omega[4][16]+fe5*omega[5][16]+fe6*omega[6][16]+fe7*omega[7][16]+fe8*omega[8][16]+fe9*omega[9][16]+fe10*omega[10][16]+fe11*omega[11][16]+fe12*omega[12][16]+fe13*omega[13][16]+fe14*omega[14][16]+fe15*omega[15][16]+fe16*omega[16][16]+fe17*omega[17][16]+fe18*omega[18][16]);

    f17=f17-(fe0*omega[0][17]+fe1*omega[1][17]+fe2*omega[2][17]+fe3*omega[3][17]+fe4*omega[4][17]+fe5*omega[5][17]+fe6*omega[6][17]+fe7*omega[7][17]+fe8*omega[8][17]+fe9*omega[9][17]+fe10*omega[10][17]+fe11*omega[11][17]+fe12*omega[12][17]+fe13*omega[13][17]+fe14*omega[14][17]+fe15*omega[15][17]+fe16*omega[16][17]+fe17*omega[17][17]+fe18*omega[18][17]);

    f18=f18-(fe0*omega[0][18]+fe1*omega[1][18]+fe2*omega[2][18]+fe3*omega[3][18]+fe4*omega[4][18]+fe5*omega[5][18]+fe6*omega[6][18]+fe7*omega[7][18]+fe8*omega[8][18]+fe9*omega[9][18]+fe10*omega[10][18]+fe11*omega[11][18]+fe12*omega[12][18]+fe13*omega[13][18]+fe14*omega[14][18]+fe15*omega[15][18]+fe16*omega[16][18]+fe17*omega[17][18]+fe18*omega[18][18]);
   


    //now, everybody streams...
   
    int Z = tid/(Nx*Ny);
    int Y = (tid - Z*Nx*Ny)/Nx;
    int X = tid - Z*Nx*Ny - Y*Nx;
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







err_t jktFunction(int nlhs,mxArray * plhs[],int nrhs,mxArray * prhs[]){

  if(nrhs!=16)
    return err("Usage:D3Q19_RegVwPe_SNL_VNL_MRT_NodeList_ts(fIn,fOut,SNL,VW_nl,PE_nl,rho_out,VNL,ux_p,uy_p,uz_p,M,Nx,Ny,Nz,NodeList,numNL)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_fOut = prhs[1];
  mxArray * m_SNL=prhs[2];
  mxArray * m_VW_nl = prhs[3];
  mxArray * m_PE_nl = prhs[4];
  mxArray * m_rho_out = prhs[5];
  mxArray * m_VNL = prhs[6];
  mxArray * m_ux_p = prhs[7];
  mxArray * m_uy_p = prhs[8];
  mxArray * m_uz_p = prhs[9];
  //float omega = mxGetScalar(prhs[7]);
  mxArray * m_M = prhs[10];
  int Nx = mxGetScalar(prhs[11]);
  int Ny = mxGetScalar(prhs[12]);
  int Nz = mxGetScalar(prhs[13]);
  mxArray * m_NodeList=prhs[14];
  int numNL = mxGetScalar(prhs[15]);

  float * fIn;
  float * fOut;
  int * SNL;
  int * VW_nl;
  int * PE_nl;
  float * rho_out;
  int * VNL;
  float * ux_p;
  float * uy_p;
  float * uz_p;
  float * M;
  int * NodeList;

  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&fOut,m_fOut);
  jkt_mem((void**)&SNL,m_SNL);
  jkt_mem((void**)&VW_nl,m_VW_nl);
  jkt_mem((void**)&PE_nl,m_PE_nl);
  jkt_mem((void**)&rho_out,m_rho_out);
  jkt_mem((void**)&VNL, m_VNL);
  jkt_mem((void**)&ux_p,m_ux_p);
  jkt_mem((void**)&uy_p,m_uy_p);
  jkt_mem((void**)&uz_p,m_uz_p);
  jkt_mem((void**)&M,m_M);
  jkt_mem((void**)&NodeList,m_NodeList);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((numNL+TPB-1)/TPB,1,1);

  D3Q19_RegVwPe_SNL_VNL_MRT_NodeList_ts<<<GRIDS,BLOCKS>>>(fIn,fOut,SNL,VW_nl,PE_nl,rho_out,
						 VNL, ux_p,uy_p,uz_p,
							  M,Nx,Ny,Nz,
							  NodeList,numNL);


  return errNone;
}
