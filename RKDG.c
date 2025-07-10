#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include<stdio.h>
#include<math.h>
#include<string.h>
#include <sys/time.h>

#define Nx 200
#define k  2
#define dimPK (k+1)
#define NumGLP 5 
#define CFL 0.1
#define pi  3.14159265358979323846

typedef struct 
{
   double phig[NumGLP][dimPK],phixg[NumGLP][dimPK];    //get_basis的变量
   double phigr[1][dimPK],phigl[1][dimPK],mm[1][dimPK];

   double lambda[NumGLP],weight[NumGLP];               //get_GLP

   double bcL,bcR,hx,hx1,xa,xb,tend;                    //init_data
   double ureal[Nx][NumGLP],xc[Nx];

   double uh[Nx][dimPK];                                 //L2Pro

   double dt,t;                                          //RK3
   double uh1[Nx][dimPK],du2[Nx][dimPK],uh2[Nx][dimPK];
   double du[Nx][dimPK];

   double uhb[Nx+2][dimPK],uhG[Nx][NumGLP];              //Lh的变量
   double flat[Nx+1][2],uhR[Nx+1][1],uhL[Nx+1][1];
   //double uhx[Nx][dimPK],du1[Nx][dimPK];
   double uR,uL,alpha;
   double test[Nx][dimPK];

}global_params;

global_params params;

double func(double u) {   //定义方程类型
    return u;
}


void get_GLP()        //储存guass积分点和权重
{
    if (NumGLP == 5) {
        params.lambda[0] = -0.9061798459386639927976269;
        params.lambda[1] = -0.5384693101056830910363144;
        params.lambda[2] = 0.0;
        params.lambda[3] = 0.5384693101056830910363144;
        params.lambda[4] = 0.9061798459386639927976269;

        params.weight[0] = 0.2369268850561890875142640;
        params.weight[1] = 0.4786286704993664680412915;
        params.weight[2] = 0.5688888888888888888888889;
        params.weight[3] = 0.4786286704993664680412915;
        params.weight[4] = 0.2369268850561890875142640;
    } 
}

void get_basis() {            //储存基函数和它的偏导数
    int i;
    for (i = 0; i < NumGLP; i++) {
        params.phig[i][0] = 1.0;
        params.phig[i][1] = params.lambda[i];
        params.phig[i][2] = params.lambda[i] * params.lambda[i] - 1.0 / 3.0;

        params.phixg[i][0] = 0.0;
        params.phixg[i][1] = 1.0 / params.hx1;
        params.phixg[i][2] = 2.0 * params.lambda[i] / params.hx1;
    }

    params.phigr[0][0] = 1.0;
    params.phigr[0][1] = 1.0;
    params.phigr[0][2] = 2.0 / 3.0;

    params.phigl[0][0] = 1.0;
    params.phigl[0][1] = -1.0;
    params.phigl[0][2] = 2.0 / 3.0;

    params.mm[0][0] = 1.0;
    params.mm[0][1] = 1.0 / 3.0;
    params.mm[0][2] = 4.0 / 45.0;
}

void init_data()
{
    int i,j;
    memset(params.ureal, 0.0, sizeof(params.ureal));    //初始化ureal
    params.xa=0.0;                                    //起始点
    params.xb=2*pi;                                   //求解域长度
    params.bcL=1.0;                                  //边界条件
    params.bcR=1.0;
    params.tend=2*pi;                                //终止时间
    params.hx=(params.xb-params.xa)/Nx;              //计算网格步长
    params.hx1=0.5L*params.hx;
    for(i=0;i<Nx;i++)
    {
        params.xc[i]=params.xa+(i+1)*params.hx-params.hx1;  //记录网格点位置
    }

    for(i=0;i<Nx;i++){                   //初始化真解
        for(j=0;j<NumGLP;j++)
        {
            params.ureal[i][j]=sin(params.xc[i]+params.hx1*params.lambda[j]);
        }
    }

}

void L2pro()                      //L2投影
 {
    int i,s,i1,d;
    memset(params.uh,0.0,sizeof(params.uh));
    for(i=0;i<Nx;i++){
        for(s=0;s<dimPK;s++){
            for(i1=0;i1<NumGLP;i1++){
                params.uh[i][s]=params.uh[i][s]+0.5L*params.weight[i1]*
                params.ureal[i][i1]*params.phig[i1][s];
            }
        }
    }

    for(i=0;i<Nx;i++){
        for(d=0;d<dimPK;d++){
            params.uh[i][d]= params.uh[i][d]/params.mm[0][d];
        }
    }

 }

 void Lh(double uhx[Nx][dimPK],double du1[Nx][dimPK])
 {
  int i,j,i1;
  memset(params.uhG,0.0,sizeof(params.uhG));
  memset(params.uhR,0.0,sizeof(params.uhR));
  memset(params.uhL,0.0,sizeof(params.uhL));
  memset(params.flat,0.0,sizeof(params.flat));

  for(i=0;i<Nx;i++){      //初始du1
    for(j=0;j<dimPK;j++){
        du1[i][j]=0.0;
    }
  }

  for(i=0;i<Nx+2;i++){
      for(j=0;j<dimPK;j++){
          if(i==0||i==Nx+1)
          {
              params.uhb[i][j]=0.0;
          }
          else
          {
              params.uhb[i][j]=uhx[i-1][j];
          }
      }
  }

  if(params.bcL==1.0&&params.bcR==1.0){   //设立边界条件
    for(i=0;i<dimPK;i++)
    {
       params.uhb[0][i]=uhx[Nx-1][i];
       params.uhb[Nx+1][i]=uhx[0][i];
    }
  }

   for(i=0;i<Nx;i++){                     //计算积分区间
      for(i1=0;i1<dimPK;i1++){
          for(j=0;j<NumGLP;j++){
              params.uhG[i][j]=params.uhG[i][j]+uhx[i][i1]*params.phig[j][i1];
          }
      }
   }

   for(i=1;i<Nx;i++){
    for(j=0;j<NumGLP;j++){
        params.test[i][j]=params.uhG[i][j];
    }
   }


   for(i=0;i<Nx;i++){
      for(j=1;j<dimPK;j++){
          for(i1=0;i1<NumGLP;i1++){
              du1[i][j]=du1[i][j]+0.5*params.weight[i1]*func(params.uhG[i][i1])*params.phixg[i1][j];
          }
      }
   }


   for(i=0;i<Nx+1;i++){                 //计算通量步骤
     for(j=0;j<dimPK;j++){
        params.uhR[i][0]=params.uhR[i][0]+params.uhb[i][j]*params.phigr[0][j];
        params.uhL[i][0]=params.uhL[i][0]+params.uhb[i][j]*params.phigl[0][j];
     }
   }

   for(i=0;i<Nx+1;i++)  
   {
      params.uR=params.uhL[i][0];
      params.uL=params.uhR[i][0];
      params.alpha=1.0L;
      params.flat[i][0]=0.5L*(func(params.uR)+func(params.uL)
      -params.alpha*(params.uR-params.uL));
   }


   for(i=0;i<Nx;i++){               //组装离散式子
      for(j=0;j<dimPK;j++){
          du1[i][j]=du1[i][j]-(1.0L/params.hx)*
          ((params.phigr[0][j])*params.flat[i+1][0]-
          params.phigl[0][j]*params.flat[i][0]);
      }
   }

 
   for(i=0;i<Nx;i++){
      for(j=0;j<dimPK;j++){
          du1[i][j]=du1[i][j]/params.mm[0][j];
      }
   }
   
}

void Lh_loc(double uhx[Nx][dimPK],double du1[Nx][dimPK])
{
    int i,j,i1;
    memset(params.uhG,0.0,sizeof(params.uhG));
    memset(params.uhR,0.0,sizeof(params.uhR));
    memset(params.uhL,0.0,sizeof(params.uhL));
    memset(params.flat,0.0,sizeof(params.flat));
  
    for(i=0;i<Nx;i++){      //初始du1
      for(j=0;j<dimPK;j++){
          du1[i][j]=0.0;
      }
    }
  
    for(i=0;i<Nx+2;i++){
        for(j=0;j<dimPK;j++){
            if(i==0||i==Nx+1)
            {
                params.uhb[i][j]=0.0;
            }
            else
            {
                params.uhb[i][j]=uhx[i-1][j];
            }
        }
    }
  
    if(params.bcL==1.0&&params.bcR==1.0){   //设立边界条件
      for(i=0;i<dimPK;i++)
      {
         params.uhb[0][i]=uhx[Nx-1][i];
         params.uhb[Nx+1][i]=uhx[0][i];
      }
    }
  
     for(i=0;i<Nx;i++){                     //计算积分区间
        for(i1=0;i1<dimPK;i1++){
            for(j=0;j<NumGLP;j++){
                params.uhG[i][j]=params.uhG[i][j]+uhx[i][i1]*params.phig[j][i1];
            }
        }
     }
  

  
     for(i=0;i<Nx;i++){
        for(j=1;j<dimPK;j++){
            for(i1=0;i1<NumGLP;i1++){
                du1[i][j]=du1[i][j]+0.5*params.weight[i1]*func(params.uhG[i][i1])*params.phixg[i1][j];
            }
        }
     }
    
     for(i=0;i<Nx;i++){
        for(j=0;j<dimPK;j++){
            params.test[i][j]=du1[i][j];
        }
       }
    
  
     for(i=0;i<Nx+1;i++){                 //计算通量步骤
       for(j=0;j<dimPK;j++){
          params.uhR[i][0]=params.uhR[i][0]+params.uhb[i][j]*params.phigr[0][j];
          params.uhL[i][0]=params.uhL[i][0]+params.uhb[i+1][j]*params.phigl[0][j];
       }
     }
  
     for(i=0;i<Nx+1;i++)  
     {
        params.uR=params.uhL[i][0];
        params.uL=params.uhR[i][0];
       // params.alpha=1.0L;
        params.flat[i][0]=func(params.uR);
        params.flat[i][1]=func(params.uL);
     }
  
  
     for(i=0;i<Nx;i++){               //组装离散式子
        for(j=0;j<dimPK;j++){
            du1[i][j]=du1[i][j]-(1.0/params.hx)*
            ((params.phigr[0][j])*params.flat[i+1][1]-params.phigl[0][j]*params.flat[i][0]);
        }
     }
  
   
     for(i=0;i<Nx;i++){
        for(j=0;j<dimPK;j++){
            du1[i][j]=du1[i][j]/params.mm[0][j];
        }
     }
}

 void RK3()
 {
   int i,j;
   params.t=0.0;
   int sum=0;
   params.dt=CFL*params.hx;
   for(i=0;i<Nx;i++){
    for(j=0;j<dimPK;j++){
        params.du2[i][j]=0.0;
    }
   }

   while (params.t<params.tend){
    if(params.t+params.dt>=params.tend)
    {
       params.dt=params.tend-params.t;
       params.t=params.tend;
       sum++;
    }
    else
    {
       params.t=params.t+params.dt;
       sum++;
    }
    if(sum%10000==0){
   printf("runing time is: %f\n",params.t);
}

// if(sum==2000){
//     break;
//  }

       Lh_loc(params.uh,params.du2);
       //Lh(params.uh,params.du2);                     //RK3步骤
       for (int i = 0; i < Nx; i++) {
           for (int j = 0; j < dimPK; j++) {
               params.uh1[i][j] = params.uh[i][j] + (1.0/3.0)*params.dt * params.du2[i][j];
           }
       }
   
       Lh_loc(params.uh1,params.du2);
       //Lh(params.uh1,params.du2);
       for (int i = 0; i < Nx; i++) {
           for (int j = 0; j < dimPK; j++) {
               params.uh2[i][j] = params.uh[i][j]+(2.0/3.0)*params.dt * params.du2[i][j];
           }
       }
   
       Lh(params.uh2,params.du2);
       Lh(params.uh,params.du); 
       for (int i = 0; i < Nx; i++) {
           for (int j = 0; j < dimPK; j++) {
               params.uh[i][j] = params.uh[i][j] +(1.0/4.0)*params.dt*
               params.du[i][j] +(3.0/4.0)*params.dt * params.du2[i][j];
           }
       }

  }
  printf("%d",sum);

}

 void output()
 {
    FILE *fp;
    int i,j;

    fp = fopen("DG_convection_solution.dat", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    for (i = 0; i < Nx; i++) {
        fprintf(fp,"%d ",i);
    for (j = 0; j < dimPK; j++) {
        fprintf(fp, "%40.35f   ", params.test[i][j]);
    }
        fprintf(fp, "\n");  // 换行
    }
    fclose(fp);
 }

 void Error()
 {
    double uE[Nx][NumGLP],L2_Error;
    int i,j,i1;
    memset(uE,0.0,sizeof(uE));
    memset(params.uhG,0.0,sizeof(params.uhG));
    L2_Error=0.0L;

    for(i=0;i<Nx;i++){
        for(i1=0;i1<dimPK;i1++){
            for(j=0;j<NumGLP;j++){
                params.uhG[i][j]=params.uhG[i][j]+params.uh[i][i1]*params.phig[j][i1];
            }
        }
    }


    for(i=0;i<Nx;i++){
        for(j=0;j<NumGLP;j++){
            uE[i][j]=fabs(params.uhG[i][j]-params.ureal[i][j]);
        }
    }

   for(i=0;i<Nx;i++){
       for(j=0;j<NumGLP;j++){
          L2_Error=L2_Error+params.hx1*params.weight[j]*(uE[i][j]*uE[i][j]);
       }
   }
   L2_Error=sqrt(L2_Error);
   printf("fina L2_error is:%.15f\n",L2_Error);
 }

 double start_timer() {
    #if defined(_WIN32)
        LARGE_INTEGER freq, start;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&start);
        return (double)start.QuadPart / (double)freq.QuadPart;
    #else
        struct timeval start;
        gettimeofday(&start, NULL);
        return (double)start.tv_sec + (double)start.tv_usec * 1e-6;
    #endif
    }
    
    double stop_timer(double start_time) {
    #if defined(_WIN32)
        LARGE_INTEGER freq, end;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&end);
        double end_time = (double)end.QuadPart / (double)freq.QuadPart;
        return end_time - start_time;
    #else
        struct timeval end;
        gettimeofday(&end, NULL);
        double end_time = (double)end.tv_sec + (double)end.tv_usec * 1e-6;
        return end_time - start_time;
    #endif
    }
    

    int main() {
        double start = start_timer();
    
        // 你的数值模拟主流程
        get_GLP();
        init_data();
        get_basis();
        L2pro();
        RK3();
        output();
        Error();
    
        double elapsed = stop_timer(start);
        printf("Wall time elapsed: %.6f seconds\n", elapsed);
    
        return 0;
    }