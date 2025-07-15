#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#define PI 3.141592653589793

// 对流方程: du/dt + c * du/dx = 0
// 使用迎风差分格式（upwind scheme）
// 初值条件: u(x,0) = sin(x)
// 周期边界条件: u(0,t) = u(2π,t)

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 开始计时
    start_time = MPI_Wtime();
    
    // 参数设置
    double L = 2.0 * PI;        // 空间域长度 [0, 2π]
    double T = 2.0;               // 时间域长度
    double c = 1.0;               // 对流速度（正值，向右传播）
    int N = 200000;                  // 总网格点数
    int Nt = 2000000;                // 时间步数
    
    double dx = L / N;            // 空间步长
    double dt = T / Nt;           // 时间步长
    
    // CFL条件检查
    double CFL = c * dt / dx;
    if (rank == 0) {
        printf("=== MPI Parallel Advection Equation Solver ===\n");
        printf("Using Upwind Finite Difference Scheme\n");
        printf("Domain: [0, 2π], c = %f\n", c);
        printf("Grid points: %d, Time steps: %d\n", N, Nt);
        printf("dx = %f, dt = %f\n", dx, dt);
        printf("CFL number: %f\n", CFL);
        if (CFL > 1.0) {
            printf("Warning: CFL > 1, scheme may be unstable!\n");
        }
        printf("Number of MPI processes: %d\n\n", size);
    }
    
    // 每个进程处理的网格点数
    int local_N = N / size;
    int remainder = N % size;
    
    // 处理不能整除的情况
    if (rank < remainder) {
        local_N++;
    }
    
    // 计算每个进程的起始位置
    int start_idx = rank * (N / size) + (rank < remainder ? rank : remainder);
    
    // 分配内存（包括边界点）
    double *u_old = (double*)malloc((local_N + 2) * sizeof(double));
    double *u_new = (double*)malloc((local_N + 2) * sizeof(double));
    double *x_local = (double*)malloc(local_N * sizeof(double));
    double *u_exact = (double*)malloc(local_N * sizeof(double));
    
    // 初始化局部坐标
    for (int i = 0; i < local_N; i++) {
        x_local[i] = (start_idx + i) * dx;
    }
    
    // 设置初值条件 u(x,0) = sin(x)
    for (int i = 1; i <= local_N; i++) {
        u_old[i] = sin(x_local[i-1]);
    }
    
    // 确定邻居进程
    int left_neighbor = (rank - 1 + size) % size;
    int right_neighbor = (rank + 1) % size;
    
    if (rank == 0) {
        printf("Initial condition: u(x,0) = sin(x)\n");
        printf("Boundary condition: periodic\n");
        printf("Analytical solution: u(x,t) = sin(x - c*t)\n\n");
        printf("Starting time integration...\n");
    }
    
    // 时间步进
    for (int t = 0; t < Nt; t++) {
        // 交换边界数据（周期边界条件）
        MPI_Sendrecv(&u_old[local_N], 1, MPI_DOUBLE, right_neighbor, 0,
                     &u_old[0], 1, MPI_DOUBLE, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&u_old[1], 1, MPI_DOUBLE, left_neighbor, 1,
                     &u_old[local_N + 1], 1, MPI_DOUBLE, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 应用迎风差分格式
        // 对于 c > 0（向右传播），使用向后差分
        // du/dt + c * (u_i - u_{i-1})/dx = 0
        // u_i^{n+1} = u_i^n - c * dt/dx * (u_i^n - u_{i-1}^n)
        for (int i = 1; i <= local_N; i++) {
            if (c > 0) {
                // 向后差分（upwind for c > 0）
                u_new[i] = u_old[i] - c * dt / dx * (u_old[i] - u_old[i-1]);
            } else {
                // 向前差分（upwind for c < 0）
                u_new[i] = u_old[i] - c * dt / dx * (u_old[i+1] - u_old[i]);
            }
        }
        
        // 更新解
        double *temp = u_old;
        u_old = u_new;
        u_new = temp;
        
        // 每隔一定步数输出结果
        if (t % (Nt / 10) == 0) {
            // 收集所有进程的结果到进程0
            double *u_global = NULL;
            int *recvcounts = NULL;
            int *displs = NULL;
            
            if (rank == 0) {
                u_global = (double*)malloc(N * sizeof(double));
                recvcounts = (int*)malloc(size * sizeof(int));
                displs = (int*)malloc(size * sizeof(int));
                
                // 计算每个进程的数据大小和位移
                for (int i = 0; i < size; i++) {
                    recvcounts[i] = N / size;
                    if (i < remainder) recvcounts[i]++;
                    displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
                }
            }
            
            // 收集数据
            MPI_Gatherv(&u_old[1], local_N, MPI_DOUBLE,
                       u_global, recvcounts, displs, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);
            
            // 输出结果
            if (rank == 0) {
                double current_time = t * dt;
                printf("Time step %d (t = %.4f):\n", t, current_time);
                printf("u[0] = %.6f, u[N/4] = %.6f, u[N/2] = %.6f, u[3N/4] = %.6f\n",
                       u_global[0], u_global[N/4], u_global[N/2], u_global[3*N/4]);
                
                free(u_global);
                free(recvcounts);
                free(displs);
            }
        }
    }
    
    // 计算最终误差
    double local_error = 0.0;
    double global_error = 0.0;
    double local_l2_error = 0.0;
    double global_l2_error = 0.0;
    double local_max_error = 0.0;
    double global_max_error = 0.0;
    
    // 计算解析解在最终时间的值
    double final_time = T;
    for (int i = 0; i < local_N; i++) {
        // 周期边界条件下的解析解
        double shifted_x = x_local[i] - c * final_time;
        // 将 shifted_x 调整到 [0, 2π] 范围内
        while (shifted_x < 0) shifted_x += 2.0 * PI;
        while (shifted_x >= 2.0 * PI) shifted_x -= 2.0 * PI;
        
        u_exact[i] = sin(shifted_x);
        
        // 计算误差
        double error = fabs(u_old[i+1] - u_exact[i]);
        local_error += error;
        local_l2_error += error * error;
        if (error > local_max_error) {
            local_max_error = error;
        }
    }
    
    // 归约操作计算全局误差
    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_l2_error, &global_l2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // 结束计时
    end_time = MPI_Wtime();
    
    // 输出最终结果和性能统计
    if (rank == 0) {
        printf("\n=== Simulation Results ===\n");
        printf("Total execution time: %.6f seconds\n", end_time - start_time);
        printf("Time per time step: %.6f seconds\n", (end_time - start_time) / Nt);
        
        // 计算平均误差和L2范数误差
        double mean_error = global_error / N;
        double l2_norm_error = sqrt(global_l2_error / N);
        
        printf("\n=== Error Analysis ===\n");
        printf("Mean absolute error: %.6e\n", mean_error);
        printf("L2 norm error: %.6e\n", l2_norm_error);
        printf("Maximum absolute error: %.6e\n", global_max_error);
        printf("Relative L2 error: %.6e%%\n", l2_norm_error / sqrt(2.0) * 100);
        
        printf("\n=== Theoretical vs Numerical ===\n");
        printf("At final time t = %.4f:\n", final_time);
    
        printf("Analytical solution: u(x,t) = sin(x - %.4f)\n", c * final_time);
        printf("CFL number used: %.6f\n", CFL);
        
        if (CFL <= 1.0) {
            printf("Scheme is stable (CFL ≤ 1)\n");
        } else {
            printf("Scheme may be unstable (CFL > 1)\n");
        }
        
        // 保存最终解到文件
        double *u_final = (double*)malloc(N * sizeof(double));
        double *u_analytical = (double*)malloc(N * sizeof(double));
        int *recvcounts = (int*)malloc(size * sizeof(int));
        int *displs = (int*)malloc(size * sizeof(int));
        
        // 计算每个进程的数据大小和位移
        for (int i = 0; i < size; i++) {
            recvcounts[i] = N / size;
            if (i < remainder) recvcounts[i]++;
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
        
        // 收集最终结果
        MPI_Gatherv(&u_old[1], local_N, MPI_DOUBLE,
                   u_final, recvcounts, displs, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        // 计算全局解析解
        for (int i = 0; i < N; i++) {
            double x = i * dx;
            double shifted_x = x - c * final_time;
            while (shifted_x < 0) shifted_x += 2.0 * PI;
            while (shifted_x >= 2.0 * PI) shifted_x -= 2.0 * PI;
            u_analytical[i] = sin(shifted_x);
        }
        
        // 输出到文件
        FILE *fp = fopen("solution_comparison.dat", "w");
        if (fp) {
            fprintf(fp, "# x numerical analytical error\n");
            for (int i = 0; i < N; i++) {
                fprintf(fp, "%.6f %.6e %.6e %.6e\n", 
                       i * dx, u_final[i], u_analytical[i], 
                       fabs(u_final[i] - u_analytical[i]));
            }
            fclose(fp);
            printf("\nResults saved to 'solution_comparison.dat'\n");
        }
        
        free(u_final);
        free(u_analytical);
        free(recvcounts);
        free(displs);
    }
    
    // 清理内存
    free(u_old);
    free(u_new);
    free(x_local);
    free(u_exact);
    
    MPI_Finalize();
    return 0;
}