#include<stdio.h>
#include<math.h>
#include<mpi.h>
#include<time.h>
#include<string.h>
#include <stdlib.h>

#define Nx 100
#define k  2
#define dimPK (k+1)
#define NumGLP 5 
#define CFL 0.1
#define pi  3.14159265358979323846 

typedef struct 
{
   // ===== 基函数相关（get_basis） =====
   double phig[NumGLP][dimPK];         
   double phixg[NumGLP][dimPK];        
   double phigr[dimPK];                
   double phigl[dimPK];                
   double mm[dimPK];                   

   // ===== Gauss-Lobatto 点（get_GLP） =====
   double lambda[NumGLP];              
   double weight[NumGLP];              

   // ===== 初值与边界条件（init_data） =====
   double bcL, bcR;                    
   double hx, hx1, xa, xb, tend;       
   double xc[Nx];                      

   // ===== 时间推进（RK3） =====
   double dt, t;                       
   double alpha;                       

} global_params;

static global_params params;

static inline double func(double u) {  
    return u;  
}

void get_GLP() {
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

void init_data_local(int idx, int N, double* ureal_local, double* uh_local) {
    params.xa = 0.0;
    params.xb = 2.0 * pi;
    params.bcL = 1.0;
    params.bcR = 1.0;
    params.tend = 2.0 * pi;
    params.hx = (params.xb - params.xa) / Nx;
    params.hx1 = params.hx * 0.5;

    // 只计算本地网格中心点
    for(int i = 0; i < N; i++) {
        params.xc[idx + i] = params.xa + (idx + i + 0.5) * params.hx;
    }

    // 计算本地精确解
    for(int i = 0; i < N; i++) {
        double x_center = params.xc[idx + i];
        for(int j = 0; j < NumGLP; j++) {
            ureal_local[i * NumGLP + j] = sin(x_center + params.hx1 * params.lambda[j]);
        }
    }
}

void L2pro_local(int N, double* ureal_local, double* uh_local) {
    // 初始化本地uh
    memset(uh_local, 0, N * dimPK * sizeof(double));
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < dimPK; j++) {
            double sum = 0.0;
            double mm_inv = 1.0 / params.mm[j];
            
            for(int s = 0; s < NumGLP; s++) {
                sum += params.weight[s] * ureal_local[i * NumGLP + s] * params.phig[s][j];
            }
            uh_local[i * dimPK + j] = 0.5 * sum * mm_inv;
        }
    }
}

// 优化的Lh函数，只处理本地数据
void Lh_local(double* uh_local, double* du_local, int N, int rank, int size,
              double* left_boundary, double* right_boundary) {
    
    // 初始化du_local
    memset(du_local, 0, N * dimPK * sizeof(double));
    
    // 设置ghost cells - 使用传入的边界数据
    double uhb[N+2][dimPK];
    
    // 左边界
    for(int j = 0; j < dimPK; j++) {
        uhb[0][j] = left_boundary[j];
    }
    
    // 本地数据
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < dimPK; j++) {
            uhb[i+1][j] = uh_local[i * dimPK + j];
        }
    }
    
    // 右边界
    for(int j = 0; j < dimPK; j++) {
        uhb[N+1][j] = right_boundary[j];
    }

    
    // 计算在GL点的值
    double uhG[N][NumGLP];
    memset(uhG, 0, sizeof(uhG));
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < NumGLP; j++) {
            for(int s = 0; s < dimPK; s++) {
                uhG[i][j] += uh_local[i * dimPK + s] * params.phig[j][s];
            }
        }
    }


    // 计算体积分项
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < NumGLP; j++) {
            for(int s = 1; s < dimPK; s++) {
                du_local[i * dimPK + s] += 0.5 * params.weight[j] * 
                                          func(uhG[i][j]) * params.phixg[j][s];
            }
        }
    }
  
    // 计算边界上的左右值
    double uhR[N+1], uhL[N+1];
    memset(uhR, 0, sizeof(uhR));
    memset(uhL, 0, sizeof(uhL));
    
    for(int i = 0; i <= N; i++) {
        for(int j = 0; j < dimPK; j++) {
            uhR[i] += uhb[i][j] * params.phigr[j];
            uhL[i] += uhb[i+1][j] * params.phigl[j];
        }
    }
    
    // 计算数值通量
    params.alpha = 1.0;
    double flux[N+1];
    for(int i = 0; i <= N; i++) {
        double uR = uhL[i];
        double uL = uhR[i];
        flux[i] = 0.5 * (func(uR) + func(uL) - params.alpha * (uR - uL));
    }
    
    // 计算边界积分项并除以质量矩阵
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < dimPK; j++) {
            du_local[i * dimPK + j] -= (1.0/params.hx) * 
                (params.phigr[j] * flux[i+1] - params.phigl[j] * flux[i]);
            du_local[i * dimPK + j] /= params.mm[j];
        }
    }

}

// 大幅优化的RK3函数
void RK3_optimized(int N, int rank, int size, double* uh_local, double* ureal_local) {
    params.t = 0.0;
    int sum = 0;
    params.dt = CFL * params.hx;
    
    // 本地工作数组
    double* uh1_local = malloc(N * dimPK * sizeof(double));
    double* uh2_local = malloc(N * dimPK * sizeof(double));
    double* du_local = malloc(N * dimPK * sizeof(double));
    
    // 边界通信缓冲区
    double left_boundary[dimPK], right_boundary[dimPK];
    
    // 邻居进程ID
    int left_neighbor = (rank == 0) ? size - 1 : rank - 1;
    int right_neighbor = (rank == size - 1) ? 0 : rank + 1;
    
    // 创建持久通信请求
    MPI_Request send_req[2], recv_req[2];
    
    while (params.t < params.tend) {
          // if(sum==1)break;
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
        
        // RK3的三个stage
        for(int stage = 0; stage < 3; stage++) {
            double *current_uh = (stage == 0) ? uh_local : 
                                ((stage == 1) ? uh1_local : uh2_local);
            
            // 非阻塞通信边界数据
            if(size > 1) {
                MPI_Irecv(left_boundary, dimPK, MPI_DOUBLE, left_neighbor, 0, 
                         MPI_COMM_WORLD, &recv_req[0]);
                MPI_Irecv(right_boundary, dimPK, MPI_DOUBLE, right_neighbor, 1, 
                         MPI_COMM_WORLD, &recv_req[1]);
                
                MPI_Isend(&current_uh[(N-1) * dimPK], dimPK, MPI_DOUBLE, right_neighbor, 0, 
                         MPI_COMM_WORLD, &send_req[0]);
                MPI_Isend(current_uh, dimPK, MPI_DOUBLE, left_neighbor, 1, 
                         MPI_COMM_WORLD, &send_req[1]);
                
                MPI_Waitall(2, recv_req, MPI_STATUSES_IGNORE);
                MPI_Waitall(2, send_req, MPI_STATUSES_IGNORE);
            } else {
                // 单进程周期边界条件
                memcpy(left_boundary, &current_uh[(N-1) * dimPK], dimPK * sizeof(double));
                memcpy(right_boundary, current_uh, dimPK * sizeof(double));
            }
            
            // 计算右端项
            Lh_local(current_uh, du_local, N, rank, size, left_boundary, right_boundary);
            
            // RK更新
            if(stage == 0) {
                for(int i = 0; i < N * dimPK; i++) {
                    uh1_local[i] = uh_local[i] + params.dt * du_local[i];
                }

            } else if(stage == 1) {
                for(int i = 0; i < N * dimPK; i++) {
                    uh2_local[i] = 0.75 * uh_local[i] + 0.25 * uh1_local[i] + 
                                   0.25 * params.dt * du_local[i];
                }
            } else {
                for(int i = 0; i < N * dimPK; i++) {
                    uh_local[i] = (1.0/3.0) * uh_local[i] + (2.0/3.0) * uh2_local[i] + 
                                  (2.0/3.0) * params.dt * du_local[i];
                }

//             for(int i = 0; i < N; i++) {
//            //for(int j = 0; j < NumGLP; j++) {
//            for(int s = 0; s < dimPK; s++) {
//                 printf("%f  ",uh_local[i*dimPK+s]);
//           // }
//         }
//            printf("rank is=%d\n",rank);
//    }

            }
        }
    }
    
    if(rank == 0) {
        printf("Total time steps: %d\n", sum);
    }
    
    // 清理
    free(uh1_local);
    free(uh2_local);
    free(du_local);
}

void Error_parallel(int N, int rank, int size, double* uh_local, double* ureal_local) {
    // 本地误差计算
    double local_uhG[N * NumGLP];
    double local_uE[N * NumGLP];
    
    // 将模态系数转换为GL点值
    memset(local_uhG, 0, sizeof(local_uhG));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < NumGLP; j++) {
            for(int s = 0; s < dimPK; s++) {
                local_uhG[i * NumGLP + j] += uh_local[i * dimPK + s] * params.phig[j][s];
            }
        }
    }
    
    // 计算本地误差
    double local_L2 = 0.0, local_L1 = 0.0, local_Linf = 0.0;
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < NumGLP; j++) {
            int idx = i * NumGLP + j;
            local_uE[idx] = fabs(local_uhG[idx] - ureal_local[idx]);
            
            double weight = params.hx1 * params.weight[j];
            local_L2 += weight * local_uE[idx] * local_uE[idx];
            local_L1 += weight * local_uE[idx];
            
            if(local_uE[idx] > local_Linf) {
                local_Linf = local_uE[idx];
            }
        }
    }
    
    // 全局归约
    double global_L2, global_L1, global_Linf;
    MPI_Reduce(&local_L2, &global_L2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_L1, &global_L1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Linf, &global_Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        global_L2 = sqrt(global_L2);
        printf("Final L2 error   = %.15f\n", global_L2);
        printf("Final L1 error   = %.15f\n", global_L1);
        printf("Final Linf error = %.15f\n", global_Linf);
    }
}

void output_parallel(int N, int rank, int size, double* uh_local) {
    if(rank == 0) {
        FILE *fp = fopen("DG_MPI_convection_solution.dat", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            return;
        }
        
        // 处理rank 0的数据
        for(int i = 0; i < N; i++) {
            fprintf(fp, "%d ", i);
            double uhG[NumGLP] = {0};
            for(int j = 0; j < NumGLP; j++) {
                for(int s = 0; s < dimPK; s++) {
                    uhG[j] += uh_local[i * dimPK + s] * params.phig[j][s];
                }
                fprintf(fp, "%.15e ", uhG[j]);
            }
            fprintf(fp, "\n");
        }
        
        // 接收并处理其他进程的数据
        for(int p = 1; p < size; p++) {
            int recv_N = Nx / size;
            if(p < (Nx % size)) recv_N++;
            
            double* recv_uh = malloc(recv_N * dimPK * sizeof(double));
            MPI_Recv(recv_uh, recv_N * dimPK, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int start_idx = p * (Nx / size) + (p < (Nx % size) ? p : (Nx % size));
            
            for(int i = 0; i < recv_N; i++) {
                fprintf(fp, "%d ", start_idx + i);
                double uhG[NumGLP] = {0};
                for(int j = 0; j < NumGLP; j++) {
                    for(int s = 0; s < dimPK; s++) {
                        uhG[j] += recv_uh[i * dimPK + s] * params.phig[j][s];
                    }
                    fprintf(fp, "%.15e ", uhG[j]);
                }
                fprintf(fp, "\n");
            }
            free(recv_uh);
        }
        fclose(fp);
    } else {
        MPI_Send(uh_local, N * dimPK, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double t_start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 负载均衡的网格分布
    int local_N = Nx / size;
    int remainder = Nx % size;
    if (rank < remainder) {
        local_N++; 
    }
    int start_idx = rank * (Nx / size) + (rank < remainder ? rank : remainder);

    // 分配本地数组
    double* uh_local = malloc(local_N * dimPK * sizeof(double));
    double* ureal_local = malloc(local_N * NumGLP * sizeof(double));

    // 初始化
    get_GLP();
    init_data_local(start_idx, local_N, ureal_local, uh_local);
    get_basis();
    L2pro_local(local_N, ureal_local, uh_local);
    
    // 时间推进
    RK3_optimized(local_N, rank, size, uh_local, ureal_local);

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;
    
    // 误差分析和输出
    Error_parallel(local_N, rank, size, uh_local, ureal_local);
    output_parallel(local_N, rank, size, uh_local);
    
    // 输出性能信息
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        printf("Wall time elapsed: %.6f seconds\n", max_elapsed);
    }

    // 清理
    free(uh_local);
    free(ureal_local);
    
    MPI_Finalize();
    return 0;
}