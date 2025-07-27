#include<stdio.h>
#include<math.h>
#include<string.h>
#include<time.h>
#include<omp.h>  // OpenMP头文件

#define Nx 5000
#define k  2
#define dimPK (k+1)
#define NumGLP 5 
#define CFL 0.2
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

   double uhb[Nx+2][dimPK],uhG[Nx][NumGLP];              //Lh的变量
   double flat[Nx+1][1],uhR[Nx+1][1],uhL[Nx+1][1];
   double uR,uL,alpha;
   double test[Nx][NumGLP];

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
    
    #pragma omp parallel for private(i)
    for(i=0;i<Nx;i++)
    {
        params.xc[i]=params.xa+(i+1)*params.hx-params.hx1;  //记录网格点位置
    }

    #pragma omp parallel for private(i,j)
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
    
    #pragma omp parallel for private(i,s,i1) collapse(2)
    for(i=0;i<Nx;i++){
        for(s=0;s<dimPK;s++){
            for(i1=0;i1<NumGLP;i1++){
                #pragma omp atomic
                params.uh[i][s] += 0.5L*params.weight[i1]*
                params.ureal[i][i1]*params.phig[i1][s];
            }
        }
    }

    #pragma omp parallel for private(i,d) collapse(2)
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

    // 初始化du1 - 并行化
    #pragma omp parallel for private(i,j) collapse(2)
    for(i=0;i<Nx;i++){
        for(j=0;j<dimPK;j++){
            du1[i][j]=0.0;
        }
    }

    // 设置边界数组 - 并行化
    #pragma omp parallel for private(i,j) collapse(2)
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

    // 设立边界条件
    if(params.bcL==1.0&&params.bcR==1.0){
        #pragma omp parallel for private(i)
        for(i=0;i<dimPK;i++)
        {
            params.uhb[0][i]=uhx[Nx-1][i];
            params.uhb[Nx+1][i]=uhx[0][i];
        }
    }

    // 计算积分区间 - 并行化
    #pragma omp parallel for private(i,i1,j)
    for(i=0;i<Nx;i++){
        for(i1=0;i1<dimPK;i1++){
            for(j=0;j<NumGLP;j++){
                #pragma omp atomic
                params.uhG[i][j] += uhx[i][i1]*params.phig[j][i1];
            }
        }
    }

    // 复制测试数据 - 并行化
    #pragma omp parallel for private(i,j) collapse(2)
    for(i=1;i<Nx;i++){
        for(j=0;j<NumGLP;j++){
            params.test[i][j]=params.uhG[i][j];
        }
    }

    // 计算体积积分项 - 并行化
    #pragma omp parallel for private(i,j,i1)
    for(i=0;i<Nx;i++){
        for(j=1;j<dimPK;j++){
            for(i1=0;i1<NumGLP;i1++){
                #pragma omp atomic
                du1[i][j] += 0.5*params.weight[i1]*func(params.uhG[i][i1])*params.phixg[i1][j];
            }
        }
    }

    // 计算通量步骤 - 此部分需要串行计算通量，但可以并行计算左右状态
    #pragma omp parallel for private(i,j)
    for(i=0;i<Nx+1;i++){
        double uhR_local = 0.0, uhL_local = 0.0;
        for(j=0;j<dimPK;j++){
            uhR_local += params.uhb[i][j]*params.phigr[0][j];
            uhL_local += params.uhb[i+1][j]*params.phigl[0][j];
        }
        params.uhR[i][0] = uhR_local;
        params.uhL[i][0] = uhL_local;
    }

    // 计算数值通量 - 并行化
    #pragma omp parallel for private(i)
    for(i=0;i<Nx+1;i++)  
    {
        double uR_local = params.uhL[i][0];
        double uL_local = params.uhR[i][0];
        double alpha_local = 1.0L;
        params.flat[i][0] = 0.5L*(func(uR_local)+func(uL_local)
        -alpha_local*(uR_local-uL_local));
    }

    // 组装离散式子 - 并行化
    #pragma omp parallel for private(i,j) collapse(2)
    for(i=0;i<Nx;i++){
        for(j=0;j<dimPK;j++){
            du1[i][j] -= (1.0L/params.hx)*
            ((params.phigr[0][j])*params.flat[i+1][0]-
            params.phigl[0][j]*params.flat[i][0]);
        }
    }
 
    // 质量矩阵求逆 - 并行化
    #pragma omp parallel for private(i,j) collapse(2)
    for(i=0;i<Nx;i++){
        for(j=0;j<dimPK;j++){
            du1[i][j] = du1[i][j]/params.mm[0][j];
        }
    }
}

void RK3()
{
    int i,j;
    params.t=0.0;
    int sum=0;
    params.dt=CFL*params.hx;
    
    #pragma omp parallel for private(i,j) collapse(2)
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
        if(sum%100==0){
            printf("running time is: %f\n",params.t);
        }

        // RK3第一步
        Lh(params.uh,params.du2);
        #pragma omp parallel for private(i,j) collapse(2)
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < dimPK; j++) {
                params.uh1[i][j] = params.uh[i][j] + params.dt * params.du2[i][j];
            }
        }

        // RK3第二步
        Lh(params.uh1,params.du2);
        #pragma omp parallel for private(i,j) collapse(2)
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < dimPK; j++) {
                params.uh2[i][j] = (3.0L/4.0L)*params.uh[i][j] +(1.0L/4.0L)*
                params.uh1[i][j] +(1.0L/4.0L)*params.dt * params.du2[i][j];
            }
        }

        // RK3第三步
        Lh(params.uh2,params.du2);
        #pragma omp parallel for private(i,j) collapse(2)
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < dimPK; j++) {
                params.uh[i][j] = (1.0L/3.0L)*params.uh[i][j] +(2.0L/3.0L)*
                params.uh2[i][j] +(2.0L/3.0L)*params.dt * params.du2[i][j];
            }
        }
    }
    printf("%d\n",sum);
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
        for (j = 0; j < NumGLP; j++) {
            fprintf(fp, "%40.35f   ", params.uh[i][j]);
        }
        fprintf(fp, "\n");
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

    // 重构数值解 - 并行化
    #pragma omp parallel for private(i,i1,j)
    for(i=0;i<Nx;i++){
        for(i1=0;i1<dimPK;i1++){
            for(j=0;j<NumGLP;j++){
                #pragma omp atomic
                params.uhG[i][j] += params.uh[i][i1]*params.phig[j][i1];
            }
        }
    }

    // 计算误差 - 并行化
    #pragma omp parallel for private(i,j) collapse(2)
    for(i=0;i<Nx;i++){
        for(j=0;j<NumGLP;j++){
            uE[i][j]=fabs(params.uhG[i][j]-params.ureal[i][j]);
        }
    }

    // L2误差计算 - 使用reduction
    #pragma omp parallel for private(i,j) reduction(+:L2_Error)
    for(i=0;i<Nx;i++){
        for(j=0;j<NumGLP;j++){
            L2_Error += params.hx1*params.weight[j]*(uE[i][j]*uE[i][j]);
        }
    }
    L2_Error=sqrt(L2_Error);
    printf("final L2_error is:%.15f\n",L2_Error);
}

int main()
{
    clock_t start_cpu, end_cpu;
    double cpu_time_used, wall_start, wall_end, wall_time_used;

    // 获取可用的线程数
    int num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", num_threads);

    // 记录 CPU 开始时间和墙钟时间
    start_cpu = clock();
    wall_start = omp_get_wtime();

    // 运行主要计算函数
    get_GLP();
    init_data();
    get_basis();
    L2pro();
    RK3();
    output();
    Error();

    // 记录结束时间
    end_cpu = clock();
    wall_end = omp_get_wtime();

    // 计算时间
    cpu_time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    wall_time_used = wall_end - wall_start;

    // 输出结果
    printf("CPU time used: %f seconds\n", cpu_time_used);
    printf("Wall clock time: %f seconds\n", wall_time_used);
    printf("Parallel efficiency: %.2f%%\n", (cpu_time_used/wall_time_used)/num_threads*100);

    return 0;
}