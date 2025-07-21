#include<stdio.h>
#include<math.h>
#include<mpi.h>
#include<time.h>
#include<string.h>

#define Nx 800
#define k  2
#define dimPK (k+1)
#define NumGLP 5 
#define CFL 0.1
#define pi  3.14159265358979323846 

typedef struct 
{
   // ===== 基函数相关（get_basis） =====
   double phig[NumGLP][dimPK];         // 插值基函数在GL点的值
   double phixg[NumGLP][dimPK];        // 插值基函数在GL点的导数
   double phigr[dimPK];                // 右端点基函数 
   double phigl[dimPK];                // 左端点基函数
   double mm[dimPK];                   // 质量矩阵

   // ===== Gauss-Lobatto 点（get_GLP） =====
   double lambda[NumGLP];              // GL点坐标
   double weight[NumGLP];              // 积分权重

   // ===== 初值与边界条件（init_data） =====
   double bcL, bcR;                    // 边界条件
   double hx, hx1, xa, xb, tend;       // 网格尺寸与区间信息
   double ureal[Nx][NumGLP];           // 精确解（用于误差分析）
   double xc[Nx];                      // 单元中心点坐标

   // ===== L2 投影（L2Pro） =====
   double uh[Nx][dimPK];               // 有限元解的系数

   // ===== 时间推进（RK3） =====
   double dt, t;                       // 时间步长、当前时间
   double uh1[Nx][dimPK];              // RK stage 1
   double du2[Nx][dimPK];              // RK stage 2
   double uh2[Nx][dimPK];              // RK stage 3
   double du[Nx][dimPK];               // 通用中间变量

   // ===== 数值通量与限制器（Lh 模块） =====
   double uhb[Nx+2][dimPK];            // 带 ghost cell 的 uh
   double uhG[Nx][NumGLP];             // uh 在 GL 点的值
   double flat[Nx+1][2];               // 限制器指标
   double uhR[Nx+1][1];                // 单元右端值
   double uhL[Nx+1][1];                // 单元左端值
   double uR, uL, alpha;               // Riemann 解相关参数
   double test[Nx][dimPK];             // 调试变量（建议后期删除）

} global_params;

// 使用static避免栈溢出
static global_params params;

static inline double func(double u) {  // 内联函数提高性能
    return u;  
}

void get_GLP() {
    // 使用常量数组初始化，避免重复计算
    if (NumGLP == 5) {
        static const double lambda_vals[5] = {
            -0.9061798459386639927976269,
            -0.5384693101056830910363144,
            0.0,                                                                                                                                                                                                                                                                                                                                                                      
            0.5384693101056830910363144,
            0.9061798459386639927976269
        };
        
        static const double weight_vals[5] = {
            0.2369268850561890875142640,
            0.4786286704993664680412915,
            0.5688888888888888888888889,
            0.4786286704993664680412915,
            0.2369268850561890875142640
        };
        
        memcpy(params.lambda, lambda_vals, sizeof(lambda_vals));
        memcpy(params.weight, weight_vals, sizeof(weight_vals));
    } 
}

void get_basis() {
    if(k == 2) {
        // 预计算常数，避免重复计算
        const double inv_hx1 = 1.0 / params.hx1;
        const double two_inv_hx1 = 2.0 * inv_hx1;
        
        for (int i = 0; i < NumGLP; i++) {      
            double lambda_i = params.lambda[i];
            double lambda_i2 = lambda_i * lambda_i;
            
            params.phig[i][0] = 1.0;
            params.phig[i][1] = lambda_i;
            params.phig[i][2] = lambda_i2 - 1.0/3.0;

            params.phixg[i][0] = 0.0;
            params.phixg[i][1] = inv_hx1;
            params.phixg[i][2] = two_inv_hx1 * lambda_i;
        }

        // 去掉多余的数组维度
        params.phigr[0] = 1.0;
        params.phigr[1] = 1.0;
        params.phigr[2] = 2.0/3.0;

        params.phigl[0] = 1.0;
        params.phigl[1] = -1.0;
        params.phigl[2] = 2.0/3.0;

        params.mm[0] = 1.0;
        params.mm[1] = 1.0/3.0;
        params.mm[2] = 4.0/45.0;
    }
}

void init_data(int idx, int N) {
    // 只初始化需要的数组部分，而不是全部清零
    for (int i = idx; i < idx + N; i++) {
        for (int j = 0; j < NumGLP; j++) {
            params.ureal[i][j] = 0.0;
        }
    }
    
    params.xa = 0.0;
    params.xb = 2.0 * pi;
    params.bcL = 1.0;
    params.bcR = 1.0;
    params.tend = 2.0 * pi;
    params.hx = (params.xb - params.xa) / Nx;
    params.hx1 = params.hx * 0.5;  // 乘法比除法快

    for(int i = 0; i < N; i++) {
        params.xc[idx + i] = params.xa + (idx + i + 0.5) * params.hx;  // 优化计算
    }

    for(int i = 0; i < N; i++) {
        double x_center = params.xc[idx + i];
        for(int j = 0; j < NumGLP; j++) {
            params.ureal[idx + i][j] = sin(x_center + params.hx1 * params.lambda[j]);
        }
    }
}

void L2pro(int idx, int N) {
    // 只清零需要的部分
    for(int i = idx; i < idx + N; i++) {
        for(int j = 0; j < dimPK; j++) {
            params.uh[i][j] = 0.0;
        }
    }
    
    // 优化循环顺序和减少数组访问
    for(int i = 0; i < N; i++) {
        int global_i = idx + i;
        for(int j = 0; j < dimPK; j++) {
            double sum = 0.0;
            double mm_inv = 1.0 / params.mm[j];  // 预计算倒数
            
            for(int s = 0; s < NumGLP; s++) {
                sum += params.weight[s] * params.ureal[global_i][s] * params.phig[s][j];
            }
            params.uh[global_i][j] = 0.5 * sum * mm_inv;
        }
    }
}

void output() {
    FILE *fp;
    int i, j;

    fp = fopen("DG_MPI_convection_solution.dat", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    for (i = 0; i < Nx; i++) {
        fprintf(fp, "%d ", i);
        for (j = 0; j <NumGLP; j++) {
            fprintf(fp, "%.15e ", params.uhG[i][j]);  // 使用科学计数法，减少精度
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// 改进后的Lh函数
void Lh(double uhx[Nx][dimPK], double du1[Nx][dimPK], int idx, int N, int rank, int size) {
    int i, j, s;
    
    // 初始化du1数组
    for(i = 0; i < Nx; i++) {
        for(j = 0; j < dimPK; j++) {
            du1[i][j] = 0.0;
        }
    }
    
    // 局部数组定义
    double local_uhG[N][NumGLP];
    double local_uhR[N+1][1];
    double local_uhL[N+1][1];
    double local_flat[N+1][2];
    
    // 初始化局部数组
    for(i = 0; i < N; i++) {
        for(j = 0; j < NumGLP; j++) {
            local_uhG[i][j] = 0.0;
        }
    }
    
    for(i = 0; i <= N; i++) {
        local_uhR[i][0] = 0.0;
        local_uhL[i][0] = 0.0;
        local_flat[i][0] = 0.0;
        local_flat[i][1] = 0.0;
    }
    
    // 设置周期边界条件 - 需要MPI通信
    double left_boundary[dimPK], right_boundary[dimPK];
    
    // 初始化边界数据
    for(i = 0; i < dimPK; i++) {
        left_boundary[i] = 0.0;
        right_boundary[i] = 0.0;
    }
    
    // 相邻进程间的通信
    if(size > 1) {
        // 向右发送，从左接收
        int left_neighbor = (rank == 0) ? size - 1 : rank - 1;
        int right_neighbor = (rank == size - 1) ? 0 : rank + 1;
        
        // 使用MPI_Sendrecv避免死锁
        MPI_Sendrecv(uhx[idx+N-1], dimPK, MPI_DOUBLE, right_neighbor, 0,
                     left_boundary, dimPK, MPI_DOUBLE, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     
        MPI_Sendrecv(uhx[idx], dimPK, MPI_DOUBLE, left_neighbor, 1,
                     right_boundary, dimPK, MPI_DOUBLE, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // 单进程情况，直接使用周期边界条件
        for(i = 0; i < dimPK; i++) {
            left_boundary[i] = uhx[Nx-1][i];
            right_boundary[i] = uhx[0][i];
        }
    }
    
    // 设置ghost cells
    for(i = 0; i < dimPK; i++) {
        params.uhb[0][i] = left_boundary[i];
        params.uhb[N+1][i] = right_boundary[i];
    }
    
    // 设置本地数据
    for(i = 0; i < N; i++) {
        for(j = 0; j < dimPK; j++) {
            params.uhb[i+1][j] = uhx[idx+i][j];
        }
    }
    
    // 计算在GL点的值
    for(i = 0; i < N; i++) {
        for(j = 0; j < NumGLP; j++) {
            for(s = 0; s < dimPK; s++) {
                local_uhG[i][j] += uhx[idx+i][s] * params.phig[j][s];
            }
        }
    }
    
    // 计算体积分项
    for(i = 0; i < N; i++) {
        for(j = 0; j < NumGLP; j++) {
            for(s = 1; s < dimPK; s++) {
                du1[idx+i][s] += 0.5 * params.weight[j] * func(local_uhG[i][j]) * params.phixg[j][s];
            }
        }
    }
    
    // 计算边界上的左右值
    for(i = 0; i <= N; i++) {
        for(j = 0; j < dimPK; j++) {
            local_uhR[i][0] += params.uhb[i][j] * params.phigr[j];
            local_uhL[i][0] += params.uhb[i+1][j] * params.phigl[j];
        }
    }
    
    // 计算数值通量
    params.alpha = 1.0; // 设置Lax-Friedrichs参数
    for(i = 0; i <= N; i++) {
        params.uR = local_uhL[i][0];
        params.uL = local_uhR[i][0];
        local_flat[i][0] = 0.5 * (func(params.uR) + func(params.uL) - params.alpha * (params.uR - params.uL));
    }
    
    // 计算边界积分项
    for(i = 0; i < N; i++) {
        for(j = 0; j < dimPK; j++) {
            du1[idx+i][j] -= (1.0/params.hx) * 
                (params.phigr[j] * local_flat[i+1][0] - params.phigl[j] * local_flat[i][0]);
        }
    }
    
    // 除以质量矩阵
    for(i = 0; i < N; i++) {
        for(j = 0; j < dimPK; j++) {
            du1[idx+i][j] /= params.mm[j];
        }
    }
}

// 改进后的RK3函数
void RK3(int idx, int N, int rank, int size) {
    int i, j;
    params.t = 0.0;
    int sum = 0;
    params.dt = CFL * params.hx;
    
    // 初始化du2数组
    for(i = 0; i < Nx; i++) {
        for(j = 0; j < dimPK; j++) {
            params.du2[i][j] = 0.0;
        }
    }
    
  
    while (params.t < params.tend) {
        if(params.t + params.dt >= params.tend) {
            params.dt = params.tend - params.t;
            params.t = params.tend;
            sum++;
        } else {
            params.t += params.dt;
            sum++;
        }
        
        if (rank == 0 && sum % 10000 == 0) {
            printf("Running time is: %f\n", params.t);
        }
        
        // RK3 第一步
        Lh(params.uh, params.du2, idx, N, rank, size);
        
        // 同步所有进程的du2数据
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                      params.du2, N * dimPK, MPI_DOUBLE, MPI_COMM_WORLD);
        
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < dimPK; j++) {
                params.uh1[i][j] = params.uh[i][j] + params.dt * params.du2[i][j];
            }
        }
        
        // RK3 第二步
        Lh(params.uh1, params.du2, idx, N, rank, size);
        
        // 同步所有进程的du2数据
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                      params.du2, N * dimPK, MPI_DOUBLE, MPI_COMM_WORLD);
        
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < dimPK; j++) {
                params.uh2[i][j] = (3.0/4.0) * params.uh[i][j] + (1.0/4.0) * params.uh1[i][j] + 
                                   (1.0/4.0) * params.dt * params.du2[i][j];
            }
        }
        
        // RK3 第三步
        Lh(params.uh2, params.du2, idx, N, rank, size);
        
        // 同步所有进程的du2数据
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                      params.du2, N * dimPK, MPI_DOUBLE, MPI_COMM_WORLD);
        
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < dimPK; j++) {
                params.uh[i][j] = (1.0/3.0) * params.uh[i][j] + (2.0/3.0) * params.uh2[i][j] + 
                                  (2.0/3.0) * params.dt * params.du2[i][j];
            }
        }

    }
    
    if(rank == 0) {
        printf("Total time steps: %d\n", sum);
    }
}

void Error()
{
    double uE[Nx][NumGLP];
    double L2_Error = 0.0, L1_Error = 0.0, Linf_Error = 0.0;
    int i, j, i1;

    // 初始化
    memset(uE, 0, sizeof(uE));
    memset(params.uhG, 0, sizeof(params.uhG));

    // Step 1: 将模态系数uh转换为Gauss点上的值uhG
    for (i = 0; i < Nx; i++) {
        for (i1 = 0; i1 < dimPK; i1++) {
            for (j = 0; j < NumGLP; j++) {
                params.uhG[i][j] += params.uh[i][i1] * params.phig[j][i1];
            }
        }
    }

    // Step 2: 计算误差uE = |uhG - ureal|
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < NumGLP; j++) {
            uE[i][j] = fabs(params.uhG[i][j] - params.ureal[i][j]);
        }
    }

    // Step 3: 计算L2误差和L1误差（加权积分）
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < NumGLP; j++) {
            double weight = params.hx1 * params.weight[j];
            L2_Error += weight * uE[i][j] * uE[i][j];
            L1_Error += weight * uE[i][j];
        }
    }
    L2_Error = sqrt(L2_Error);

    // Step 4: 计算L∞误差（所有Gauss点的最大误差）
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < NumGLP; j++) {
            if (uE[i][j] > Linf_Error) {
                Linf_Error = uE[i][j];
            }
        }
    }

    // Step 5: 输出结果
    printf("Final L2 error   = %.15f\n", L2_Error);
    printf("Final L1 error   = %.15f\n", L1_Error);
    printf("Final Linf error = %.15f\n", Linf_Error);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double t_start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_N = Nx / size;
    int remainder = Nx % size;

    if (rank < remainder) {
        local_N++; 
    }
    int start_idx = rank * (Nx / size) + (rank < remainder ? rank : remainder);

    get_GLP();
    init_data(start_idx, local_N);
    get_basis();
    L2pro(start_idx, local_N);
    
    // 传递必要的参数
    RK3(start_idx, local_N, rank, size);

       double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    
    // 收集所有进程的结果到rank 0
    if(rank == 0) {
        // 收集其他进程的数据
        for(int p = 1; p < size; p++) {
            int recv_N = Nx / size;
            if(p < remainder) recv_N++;
            int recv_idx = p * (Nx / size) + (p < remainder ? p : remainder);
            
            MPI_Recv(&params.uh[recv_idx], recv_N * dimPK, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&params.ureal[recv_idx], recv_N * NumGLP, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        Error();
        output();
    } else {
        // 发送数据到rank 0
        MPI_Send(&params.uh[start_idx], local_N * dimPK, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&params.ureal[start_idx], local_N * NumGLP, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    if(rank == 0) {
        printf("Wall time elapsed: %.6f seconds\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}