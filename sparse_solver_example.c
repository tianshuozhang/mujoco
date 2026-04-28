#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mujoco/mujoco.h>

// 稀疏矩阵结构
typedef struct {
    int n;              // 矩阵维度
    int* rownnz;        // 每行的非零元素数量
    int* rowadr;        // 每行的起始地址
    int* colind;        // 列索引
    mjtNum* mat;        // 矩阵元素
} SparseMatrix;

// 创建稀疏矩阵
SparseMatrix* create_sparse_matrix(int n, int nnz) {
    SparseMatrix* sp = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    sp->n = n;
    sp->rownnz = (int*)calloc(n, sizeof(int));
    sp->rowadr = (int*)malloc(n * sizeof(int));
    sp->colind = (int*)malloc(nnz * sizeof(int));
    sp->mat = (mjtNum*)malloc(nnz * sizeof(mjtNum));
    return sp;
}

// 释放稀疏矩阵
void free_sparse_matrix(SparseMatrix* sp) {
    if (sp) {
        free(sp->rownnz);
        free(sp->rowadr);
        free(sp->colind);
        free(sp->mat);
        free(sp);
    }
}

// 示例：创建一个简单的稀疏矩阵
void create_example_sparse_matrix(SparseMatrix* sp) {
    int n = sp->n;
    
    // 设置每行的非零元素数量
    sp->rownnz[0] = 2;  // 对角线 + 一个非零
    sp->rownnz[1] = 3;  // 对角线 + 两个非零
    sp->rownnz[2] = 3;
    sp->rownnz[3] = 2;
    sp->rownnz[4] = 2;
    
    // 计算每行的起始地址
    sp->rowadr[0] = 0;
    for (int i = 1; i < n; i++) {
        sp->rowadr[i] = sp->rowadr[i-1] + sp->rownnz[i-1];
    }
    
    // 设置列索引和矩阵值 (创建一个对称正定矩阵)
    // 第一行
    sp->colind[0] = 0; sp->mat[0] = 4.0;
    sp->colind[1] = 1; sp->mat[1] = 1.0;
    
    // 第二行
    sp->colind[2] = 0; sp->mat[2] = 1.0;
    sp->colind[3] = 1; sp->mat[3] = 5.0;
    sp->colind[4] = 2; sp->mat[4] = 1.0;
    
    // 第三行
    sp->colind[5] = 1; sp->mat[5] = 1.0;
    sp->colind[6] = 2; sp->mat[6] = 6.0;
    sp->colind[7] = 3; sp->mat[7] = 1.0;
    
    // 第四行
    sp->colind[8] = 2; sp->mat[8] = 1.0;
    sp->colind[9] = 3; sp->mat[9] = 7.0;
    
    // 第五行
    sp->colind[10] = 3; sp->mat[10] = 1.0;
    sp->colind[11] = 4; sp->mat[11] = 8.0;
}

// 直接法：稀疏Cholesky分解求解
void solve_direct_method_cholesky(SparseMatrix* sp, mjtNum* b, mjtNum* x) {
    int n = sp->n;
    
    printf("\n=== 直接法：稀疏Cholesky分解 ===\n");
    
    // 1. 符号分解：计算稀疏模式
    int L_rownnz[n], L_rowadr[n], L_colind[100];  // 假设最多100个非零元素
    int LT_rownnz[n], LT_rowadr[n], LT_colind[100], LT_map[100];
    
    int total_nnz = mju_cholFactorSymbolic(
        L_colind, L_rownnz, L_rowadr,
        LT_colind, LT_rownnz, LT_rowadr, LT_map,
        sp->rownnz, sp->rowadr, sp->colind, n, NULL
    );
    
    printf("符号分解完成，总非零元素数: %d\n", total_nnz);
    
    // 2. 数值分解：计算Cholesky因子
    mjtNum L[100];  // 存储Cholesky因子
    
    int rank = mju_cholFactorNumeric(
        L, n, 1e-10,
        L_rownnz, L_rowadr, L_colind,
        LT_rownnz, LT_rowadr, LT_colind, LT_map,
        sp->mat,
        sp->rownnz, sp->rowadr, sp->colind,
        NULL
    );
    
    printf("数值分解完成，矩阵秩: %d\n", rank);
    
    // 3. 前向和后向代入求解
    mju_cholSolveSparse(x, L, b, n, L_rownnz, L_rowadr, L_colind);
    
    printf("直接法求解完成\n");
}

// 直接法：稀疏LU分解求解
void solve_direct_method_lu(SparseMatrix* sp, mjtNum* b, mjtNum* x) {
    int n = sp->n;
    
    printf("\n=== 直接法：稀疏LU分解 ===\n");
    
    // 1. LU分解
    mjtNum LU[100];  // 存储LU因子 (L+U-1)
    int scratch[n];
    
    mju_factorLUSparse(LU, n, scratch, sp->rownnz, sp->rowadr, sp->colind, NULL);
    
    printf("LU分解完成\n");
    
    // 2. 求解
    mju_solveLUSparse(x, LU, b, n, sp->rownnz, sp->rowadr, NULL, sp->colind, NULL);
    
    printf("直接法求解完成\n");
}

// 间接法：共轭梯度法
void solve_iterative_method_cg(SparseMatrix* sp, mjtNum* b, mjtNum* x, int max_iter) {
    printf("\n=== 间接法：共轭梯度法 ===\n");
    
    int n = sp->n;
    mjtNum r[n], p[n], Ap[n];
    mjtNum alpha, beta, r_dot, r_dot_new;
    mjtNum tol = 1e-10;
    
    // 初始化 x = 0
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }
    
    // r = b - A*x = b (因为x=0)
    for (int i = 0; i < n; i++) {
        r[i] = b[i];
    }
    
    // p = r
    for (int i = 0; i < n; i++) {
        p[i] = r[i];
    }
    
    // r_dot = r'*r
    r_dot = 0.0;
    for (int i = 0; i < n; i++) {
        r_dot += r[i] * r[i];
    }
    
    printf("初始残差范数: %.6e\n", sqrt(r_dot));
    
    // 共轭梯度迭代
    for (int iter = 0; iter < max_iter; iter++) {
        // Ap = A*p
        mju_mulMatVecSparse(Ap, sp->mat, p, n, sp->rownnz, sp->rowadr, sp->colind, NULL);
        
        // alpha = r'*r / (p'*Ap)
        mjtNum pAp = 0.0;
        for (int i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }
        alpha = r_dot / pAp;
        
        // x = x + alpha*p
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
        }
        
        // r = r - alpha*Ap
        for (int i = 0; i < n; i++) {
            r[i] -= alpha * Ap[i];
        }
        
        // r_dot_new = r'*r
        r_dot_new = 0.0;
        for (int i = 0; i < n; i++) {
            r_dot_new += r[i] * r[i];
        }
        
        printf("迭代 %d: 残差范数 = %.6e\n", iter+1, sqrt(r_dot_new));
        
        // 检查收敛
        if (sqrt(r_dot_new) < tol) {
            printf("在 %d 次迭代后收敛\n", iter+1);
            break;
        }
        
        // beta = r_dot_new / r_dot
        beta = r_dot_new / r_dot;
        
        // p = r + beta*p
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        r_dot = r_dot_new;
    }
}

// 验证解的正确性
void verify_solution(SparseMatrix* sp, mjtNum* x, mjtNum* b) {
    int n = sp->n;
    mjtNum Ax[n], residual[n];
    
    // 计算 Ax
    mju_mulMatVecSparse(Ax, sp->mat, x, n, sp->rownnz, sp->rowadr, sp->colind, NULL);
    
    // 计算残差
    mjtNum residual_norm = 0.0;
    for (int i = 0; i < n; i++) {
        residual[i] = Ax[i] - b[i];
        residual_norm += residual[i] * residual[i];
    }
    
    printf("\n=== 解的验证 ===\n");
    printf("残差范数: %.6e\n", sqrt(residual_norm));
    
    if (sqrt(residual_norm) < 1e-8) {
        printf("解是正确的！\n");
    } else {
        printf("警告：解可能有误差\n");
    }
}

// 打印向量
void print_vector(const char* name, mjtNum* vec, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f", vec[i]);
        if (i < n-1) printf(", ");
    }
    printf("]\n");
}

int main() {
    printf("MuJoCo稀疏矩阵求解器示例\n");
    printf("===========================\n");
    
    const int n = 5;
    
    // 创建稀疏矩阵
    SparseMatrix* sp = create_sparse_matrix(n, 12);
    create_example_sparse_matrix(sp);
    
    // 创建右端项
    mjtNum b[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    printf("右端项 b:\n");
    print_vector("b", b, n);
    
    // 方法1：直接法 - 稀疏Cholesky分解
    mjtNum x_cholesky[n];
    solve_direct_method_cholesky(sp, b, x_cholesky);
    print_vector("x_cholesky", x_cholesky, n);
    verify_solution(sp, x_cholesky, b);
    
    // 方法2：直接法 - 稀疏LU分解
    mjtNum x_lu[n];
    solve_direct_method_lu(sp, b, x_lu);
    print_vector("x_lu", x_lu, n);
    verify_solution(sp, x_lu, b);
    
    // 方法3：间接法 - 共轭梯度法
    mjtNum x_cg[n];
    solve_iterative_method_cg(sp, b, x_cg, 100);
    print_vector("x_cg", x_cg, n);
    verify_solution(sp, x_cg, b);
    
    // 比较三种方法的结果
    printf("\n=== 方法比较 ===\n");
    mjtNum diff_cholesky_lu = 0.0, diff_cholesky_cg = 0.0;
    for (int i = 0; i < n; i++) {
        diff_cholesky_lu += fabs(x_cholesky[i] - x_lu[i]);
        diff_cholesky_cg += fabs(x_cholesky[i] - x_cg[i]);
    }
    printf("Cholesky vs LU 差异: %.6e\n", diff_cholesky_lu);
    printf("Cholesky vs CG 差异: %.6e\n", diff_cholesky_cg);
    
    // 释放内存
    free_sparse_matrix(sp);
    
    return 0;
}