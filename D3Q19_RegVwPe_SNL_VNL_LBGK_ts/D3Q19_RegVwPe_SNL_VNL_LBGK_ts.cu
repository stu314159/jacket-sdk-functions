#include "jacketSDK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <iostream>

#define TPB 64



__global__ void D3Q19_RegVwPe_SNL_VNL_LBGK_ts(const float * fIn, float * fOut,
					      const int * SNL,
					      const int * VW_nl,
					      const int * PE_nl, 
					      const int * VNL,
					      const float * ux_p,
					      const float * uy_p,
					      const float * uz_p,
					      const float * rho_out,
					      const float omega,
					      const int Nx, const int Ny, const int Nz)
{

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  int nnodes=Nx*Ny*Nz;
  if(tid<nnodes){
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


    if(SNL[tid]==0){

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
    }else{
      //bounce back
      f0=f0-omega*(f0-fe0);
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

  if(nrhs!=14)
    return err("Usage:D3Q19_RegVwPe_SNL_VNL_LBGK_ts(fIn,fOut,SNL,VW_nl,PE_nl,VNL,ux_p,uy_p,uz_p,rho_out,omega,Nx,Ny,Nz)");

  mxArray * m_fIn = prhs[0];
  mxArray * m_fOut = prhs[1];
  mxArray * m_SNL=prhs[2];
  mxArray * m_VW_nl = prhs[3];
  mxArray * m_PE_nl = prhs[4];
  mxArray * m_VNL = prhs[5];
  mxArray * m_ux_p = prhs[6];
  mxArray * m_uy_p = prhs[7];
  mxArray * m_uz_p = prhs[8];
  mxArray * m_rho_out = prhs[9];
  float omega = mxGetScalar(prhs[10]);
  int Nx = mxGetScalar(prhs[11]);
  int Ny = mxGetScalar(prhs[12]);
  int Nz = mxGetScalar(prhs[13]);

  float * fIn;
  float * fOut;
  int * SNL;
  int * VW_nl;
  int * VNL;
  int * PE_nl;
  float * ux_p;
  float * uy_p;
  float * uz_p;
  float * rho_out;

  jkt_mem((void**)&fIn,m_fIn);
  jkt_mem((void**)&fOut,m_fOut);
  jkt_mem((void**)&SNL,m_SNL);
  jkt_mem((void**)&VW_nl,m_VW_nl);
  jkt_mem((void**)&VNL,m_VNL);
  jkt_mem((void**)&PE_nl,m_PE_nl);
  jkt_mem((void**)&ux_p,m_ux_p);
  jkt_mem((void**)&uy_p,m_uy_p);
  jkt_mem((void**)&uz_p,m_uz_p);
  jkt_mem((void**)&rho_out,m_rho_out);

  dim3 BLOCKS(TPB,1,1);
  dim3 GRIDS((Nx*Ny*Nz+TPB-1)/TPB,1,1);

  D3Q19_RegVwPe_SNL_VNL_LBGK_ts<<<GRIDS,BLOCKS>>>(fIn,fOut,SNL,VW_nl,PE_nl,VNL,ux_p,uy_p,uz_p,
						  rho_out,
						  omega,Nx,Ny,Nz);


  return errNone;
}
