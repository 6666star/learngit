#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//***********************************************************
//设定所有定值
//*********************************************************
constexpr int Nx=32;
constexpr int k=2;
constexpr int NumGLP=5;
constexpr int dimPK=(k+1);
constexpr double CFL=0.2;
constexpr double pi=3.14159265358979323846;
constexpr double xa=0.0;
constexpr double xb=(2*pi);
constexpr double bcL=1.0;
constexpr double bcR=1.0;
constexpr double tend=(2*pi);
constexpr double hx=((xb-xa)/Nx);
constexpr double hx1=(hx/2.0);
constexpr double dt=(CFL*hx);
//***********************************************************
//申明结构体，把所有数组变量存到结构体里面
//***********************************************************
typedef struct 
{
   double phig[NumGLP][dimPK],phixg[NumGLP][dimPK];    //get_basis的变量

   double ureal[Nx][NumGLP],xc[Nx];

   double uh[Nx][dimPK];                                 //L2Pro

   double uh1[Nx][dimPK],du2[Nx][dimPK],uh2[Nx][dimPK];

   double uhb[Nx+2][dimPK],uhG[Nx][NumGLP];              //Lh的变量
   double flat[Nx+1][1],uhR[Nx+1][1],uhL[Nx+1][1],flatb[Nx][1];
   double uR,uL,alpha;

   double lambda[5],weight[5];
   double phigl[1][3],phigr[1][3], mm[1][3];
}global_params;

global_params h_params;   // 主机端数据结构
global_params* d_params;  // 设备端结构体指针
//***********************************************************
//后面需要调用的函数
//***********************************************************

__host__ __device__ double func(double u);

double f(double u);

void init_data();

void L2pro();

void RK3();

void output();

void Error();

__global__ void  Lh1(double uhx[Nx][dimPK],double du1[Nx][dimPK],global_params *d_params);

__global__ void  Lh2(double uhx[Nx][dimPK],double du1[Nx][dimPK],global_params *d_params);

__global__  void  Lh_add_1(double uh_new[Nx][dimPK],double uh[Nx][dimPK],double du[Nx][dimPK]);

__global__  void  Lh_add_2(double uh_new[Nx][dimPK],double uh[Nx][dimPK],double uh1[Nx][dimPK],double du[Nx][dimPK]);

__global__  void  Lh_add_3(double uh_new[Nx][dimPK],double uh[Nx][dimPK],double uh2[Nx][dimPK],double du[Nx][dimPK]);
//***********************************************************
//申明需要调用的函数
//***********************************************************
__host__ __device__ double  func(double u)       
{
    return u;
}

double  f(double u)
{
    return sin(u);
}

void init_data()
{

    h_params.lambda[0] = -0.9061798459386639927976269;
    h_params.lambda[1] = -0.5384693101056830910363144;
    h_params.lambda[2] = 0.0;
    h_params.lambda[3] = 0.5384693101056830910363144;
    h_params.lambda[4] = 0.9061798459386639927976269;

    h_params.weight[0] = 0.2369268850561890875142640;
    h_params.weight[1] = 0.4786286704993664680412915;
    h_params.weight[2] = 0.5688888888888888888888889;
    h_params.weight[3] = 0.4786286704993664680412915;
    h_params.weight[4] = 0.2369268850561890875142640;

    h_params.phigr[0][0] = 1.0;
    h_params.phigr[0][1] = 1.0;
    h_params.phigr[0][2] = 2.0 / 3.0;

    h_params.phigl[0][0] = 1.0;
    h_params.phigl[0][1] = -1.0;
    h_params.phigl[0][2] = 2.0 / 3.0;

    h_params.mm[0][0] = 1.0;
    h_params.mm[0][1] = 1.0 / 3.0;
    h_params.mm[0][2] = 4.0 / 45.0;
    

    for (int i = 0; i < NumGLP; i++) 
    {
        h_params.phig[i][0] = 1.0;
        h_params.phig[i][1] = h_params.lambda[i];
        h_params.phig[i][2] = h_params.lambda[i] * h_params.lambda[i] - 1.0 / 3.0;
    
        h_params.phixg[i][0] = 0.0;
        h_params.phixg[i][1] = 1.0 / hx1;
        h_params.phixg[i][2] = 2.0 *h_params.lambda[i] / hx1;
    }

    for(int i=0;i<Nx;i++)
    {
        h_params.xc[i] = xa + (i + 1) * hx - hx1;  //记录网格位置
       // printf("%.10f\n",h_params.xc(i));
        
    }
   
    for(int i=0;i<Nx;i++)
    {
        for (int j = 0; j < NumGLP; j++)
        {
           h_params.ureal[i][j] = f(h_params.xc[i] + hx1 * h_params.lambda[j]);
        }       
    }


}

void L2pro()
{
    int i,s,i1,d;
    memset(h_params.uh,0.0,sizeof(h_params.uh));
    
    for(i=0;i<Nx;i++)
    {
        for(s=0;s<dimPK;s++)
        {
            for(i1=0;i1<NumGLP;i1++)
            {
                h_params.uh[i][s]=h_params.uh[i][s]+0.5*h_params.weight[i1]*
                h_params.ureal[i][i1]*h_params.phig[i1][s];
            }
        }
      
        for(d=0;d<dimPK;d++)
        {
            h_params.uh[i][d]= h_params.uh[i][d]/h_params.mm[0][d];
        }
    }  
    
    
}

void output()    //输出函数将结果输出到文本
 {
    FILE *fp;
    int i,j;

    fp = fopen("DG_convection_solution.dat", "w");
    if (fp == NULL) 
    {
        printf("Error opening file!\n");
        exit(-1);
    }


    fprintf(fp, "uhb\n"); 

    for (i = 0; i < Nx; i++)
    {
    fprintf(fp,"%d ",i);
          for (j = 0; j < dimPK; j++)
          {
            fprintf(fp, "%40.35f", h_params.uh[i][j]);
          }
    fprintf(fp, "\n");  // 换行
    }

   
    fprintf(fp, "du1\n"); 

    for (i = 0; i < Nx; i++)
    {
    fprintf(fp,"%d ",i);
          for (j = 0; j < dimPK; j++)
          {
            fprintf(fp, "%40.35f", h_params.du2[i][j]);
          }
    fprintf(fp, "\n");  // 换行
    }

    fprintf(fp, "uh2\n"); 

    for (i = 0; i < Nx; i++)
    {
    fprintf(fp,"%d ",i);
          for (j = 0; j < dimPK; j++)
          {
            fprintf(fp, "%40.35f", h_params.uh2[i][j]);
          }
    fprintf(fp, "\n");  // 换行
    }


    fclose(fp);
 }

 __global__ void  Lh1(double uhx[Nx][dimPK],double du1[Nx][dimPK],global_params *d_params)
 {
    int i,i1,j;

     i= blockIdx.x * blockDim.x + threadIdx.x;     //step0
    if(i<Nx)
        {
            for(j=0;j<dimPK;j++)
              {
                du1[i][j]=0.0;
              }

            for(j=0;j<NumGLP;j++)
              {
                 d_params->uhG[i][j]=0.0;
              }
            
        }

        i= blockIdx.x * blockDim.x + threadIdx.x;
        if(i<Nx+1)
           {
                d_params->flat[i][0]=0.0;
                d_params->uhL[i][0]=0.0;
                d_params->uhR[i][0]=0.0;
           }  
        
           
           i = blockIdx.x * blockDim.x + threadIdx.x;
           if (i < Nx + 2) {
               for (j = 0; j < dimPK; j++)
                {

                if (i == 0 || i == Nx + 1)
                {
                    d_params->uhb[0][j] = uhx[Nx - 1][j];
                    d_params->uhb[Nx + 1][j] = uhx[0][j];
                } 
                else
                {
                    d_params->uhb[i][j] =uhx[i-1][j];
                }
                   
               }
           }
           

               i= blockIdx.x * blockDim.x + threadIdx.x;
               if(i<Nx)
                {
                       for(i1=0;i1<dimPK;i1++)
                       {
                           for(j=0;j<NumGLP;j++)
                           {
                               d_params->uhG[i][j]+=uhx[i][i1]*d_params->phig[j][i1];
                           }
                       }
            
                   for(j=1;j<dimPK;j++)
                   {
                       for(i1=0;i1<NumGLP;i1++)
                       {
                           du1[i][j]+=0.5*d_params->weight[i1]*func(d_params->uhG[i][i1])*d_params->phixg[i1][j];
                       }
                   }

               }

      i= blockIdx.x * blockDim.x + threadIdx.x;     //step 3计算通量函数
      if(i<Nx+1)
         {
              for(j=0;j<dimPK;j++)
              {
                  d_params->uhR[i][0]=d_params->uhR[i][0]+d_params->uhb[i][j]*d_params->phigr[0][j];
                  d_params->uhL[i][0]=d_params->uhL[i][0]+d_params->uhb[i][j]*d_params->phigl[0][j];
              } 
              d_params->uR=d_params->uhL[i][0]; 
              d_params->uL=d_params->uhR[i][0];
              d_params->alpha=1.0;
              d_params->flat[i][0]=0.5*(func(d_params->uR)+func(d_params->uL)-d_params->alpha*(d_params->uR-d_params->uL));
           }


 }

__global__ void  Lh2(double uhx[Nx][dimPK],double du1[Nx][dimPK],global_params *d_params)
{
    int i,j;
    i= blockIdx.x * blockDim.x + threadIdx.x;
    if(i<Nx)
    {
        for(j=0;j<dimPK;j++)
        {
            du1[i][j]-=(1.0/hx)*( d_params->phigr[0][j]*d_params->flat[i+1][0]- d_params->phigl[0][j]*d_params->flat[i][0]);
        }
    
        for(j=0;j<dimPK;j++)
        {
        du1[i][j]= du1[i][j]/ d_params->mm[0][j];
        }
    }
}

__global__  void  Lh_add_1(double uh_new[Nx][dimPK],double uh[Nx][dimPK],double du[Nx][dimPK])
{
    int i,j;
    i= blockIdx.x * blockDim.x + threadIdx.x;
    if(i<Nx)
    {
        for ( j = 0; j < dimPK; j++) {
           uh_new[i][j] =uh[i][j] + dt * du[i][j];
        }
    }  
     
}

__global__  void  Lh_add_2(double uh_new[Nx][dimPK],double uh[Nx][dimPK],double uh1[Nx][dimPK],double du[Nx][dimPK])
{
    int i,j;
    i= blockIdx.x * blockDim.x + threadIdx.x;
    if(i<Nx)
    {
        for ( j = 0; j < dimPK; j++) {
           uh_new[i][j]=(3.0/4.0)*uh[i][j]+(1.0/4.0)*
            uh1[i][j]+(1.0/4.0)*dt*du[i][j];
        }
    }  
     
}
    
__global__  void  Lh_add_3(double uh_new[Nx][dimPK],double uh[Nx][dimPK],double uh2[Nx][dimPK],double du[Nx][dimPK])
{
    int i,j;
    i= blockIdx.x * blockDim.x + threadIdx.x;
    if(i<Nx)
    {
        for ( j = 0; j < dimPK; j++) {
                  uh_new[i][j] = (1.0/3.0)*uh[i][j] +(2.0/3.0)*
                 uh2[i][j] +(2.0/3.0)*dt *du[i][j];
        }
    }  
     
}

 void RK3()
 {
        double dt1=(CFL*hx);
        double t=0.0;
        dim3 block(32);
        dim3 grid((Nx + block.x - 1)/32+1);  //设置 block 和 grid
        double sum=0;
    
        memset( h_params.du2,0.0,sizeof(h_params.du2));
     
      while (t<tend)
      {
       if(t+dt1>=tend)
        {
           dt1=tend-t;
           t=tend;
           sum++;
        }
       else
        {
           sum++;
           t=t+dt1;
        } 
        if(sum==(Nx/CFL)+1)
        {
            break;
        }
        // if(sum==2)//在这里调整循环次数
        // {
        //     break;
        // }
         Lh1<<<grid,block>>>(d_params->uh,d_params->du2,d_params); 
 
         Lh2<<<grid,block>>>(d_params->uh,d_params->du2,d_params);                     //RK3-1步骤

         Lh_add_1<<<grid,block>>>(d_params->uh1,d_params->uh,d_params->du2);

  
         Lh1<<<grid,block>>>(d_params->uh1,d_params->du2,d_params);
    
         Lh2<<<grid,block>>>(d_params->uh1,d_params->du2,d_params);                     //RK3-2步骤

         Lh_add_2<<<grid,block>>>(d_params->uh2,d_params->uh,d_params->uh1,d_params->du2);


         Lh1<<<grid,block>>>(d_params->uh2,d_params->du2,d_params);

         Lh2<<<grid,block>>>(d_params->uh2,d_params->du2,d_params);     
                                                                                          //RK3-3步骤
         Lh_add_3<<<grid,block>>>(d_params->uh,d_params->uh,d_params->uh2,d_params->du2);
         
     
    }
    
      printf("sum_number is:%f\n",sum); 
      
 }

 void Error()
 {
    double uE[Nx][NumGLP],L2_Error;
    int i,j,i1;
    memset(uE,0.0,sizeof(uE));
    memset(h_params.uhG,0.0,sizeof(h_params.uhG));
    L2_Error=0.0;

    for(i=0;i<Nx;i++)
    {
        for(i1=0;i1<dimPK;i1++)
        {
            for(j=0;j<NumGLP;j++)
            {
                h_params.uhG[i][j]+=h_params.uh[i][i1]*h_params.phig[j][i1];
            }
        }
    }


    for(i=0;i<Nx;i++)
    {
        for(j=0;j<NumGLP;j++)
        {
            uE[i][j]=fabs(h_params.uhG[i][j]-h_params.ureal[i][j]);
        }
    }

   for(i=0;i<Nx;i++)
   {
       for(j=0;j<NumGLP;j++)
       {
          L2_Error=L2_Error+hx1*h_params.weight[j]*(uE[i][j]*uE[i][j]);
       }
   }
   L2_Error=sqrt(L2_Error);
   printf("fina L2_error is:%.15f\n",L2_Error);
 }

//***********************************************************
//主程序
//***********************************************************
#include <sys/time.h>

int main() {
    int iDev = 7;
    cudaSetDevice(iDev);
    printf("set GPU %d for computing.\n", iDev);
    
    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 开始系统级计时
       struct timeval cpu_start, cpu_end;
       gettimeofday(&cpu_start, NULL);
    
    // 初始化数据
    init_data();
    L2pro();
    
    // 在设备端分配结构体
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_params, sizeof(global_params));
    if (cudaStatus != cudaSuccess) {
        printf("fail to cudaMalloc\n");
        exit(-1);
    }
    
    
    // 开始 GPU 计时
    cudaEventRecord(start, 0);
    
    // 从主机端拷贝数据到设备端
    cudaStatus = cudaMemcpy(d_params, &h_params, sizeof(global_params), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("fail to cudaMemcpy from host to device\n");
        exit(-1);
    }
    
    // 启动内核函数
    RK3();   // 在RK3里面启动线程
    
    // 将数据从设备端拷贝回主机端
    cudaStatus = cudaMemcpy(&h_params, d_params, sizeof(global_params), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("fail to cudaMemcpy from device to host\n");
        exit(-1);
    }
    
    // 停止 GPU 计时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 停止系统级计时
     gettimeofday(&cpu_end, NULL);
    
    // 处理结果
    output();
    Error();
    
    // 计算系统级运行时间
    double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) + 
                      (cpu_end.tv_usec - cpu_start.tv_usec) / 1000000.0;
    printf("Wall clock time is: %.6f seconds\n", cpu_time);
    
    // 计算 GPU 运行时间
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU running time is: %.6f milliseconds\n", gpu_time);
    
    // 释放资源
    cudaFree(d_params);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}