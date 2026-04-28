# MuJoCo稀疏矩阵求解器使用指南

## 概述

MuJoCo提供了丰富的稀疏矩阵求解器，包括直接法和间接法。本示例展示了如何使用这些求解器来解决稀疏线性方程组 Ax = b。

## 可用的求解器

### 直接法

1. **稀疏Cholesky分解** (`mju_cholFactorSparse`, `mju_cholSolveSparse`)
   - 适用于对称正定矩阵
   - 分解：A = L*L'
   - 两步过程：符号分解 + 数值分解

2. **稀疏LU分解** (`mju_factorLUSparse`, `mju_solveLUSparse`)
   - 适用于一般方阵
   - 分解：A = L*U
   - 假设树状拓扑结构

### 间接法

1. **共轭梯度法** (CG)
   - 适用于对称正定矩阵
   - 迭代方法，内存效率高
   - 需要矩阵-向量乘法操作

2. **预处理共轭梯度法**
   - 使用预处理矩阵加速收敛
   - 适用于病态矩阵

## 编译说明

### 前提条件

1. 已安装CMake (>= 3.10)
2. 已编译MuJoCo库
3. C编译器 (gcc, clang, 或 MSVC)

### 编译步骤

#### Linux/macOS

```bash
# 1. 编译MuJoCo (如果还没有)
cd mujoco
mkdir build && cd build
cmake ..
cmake --build .

# 2. 编译示例程序
cd ..
gcc -o sparse_solver_example sparse_solver_example.c \
    -I./build/include -L./build -lmujoco -lm -lpthread

# 3. 运行
./sparse_solver_example
```

#### Windows (使用MinGW)

```bash
# 1. 编译MuJoCo (如果还没有)
cd mujoco
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .

# 2. 编译示例程序
cd ..
gcc -o sparse_solver_example.exe sparse_solver_example.c \
    -I./build/include -L./build -lmujoco -lm

# 3. 运行
sparse_solver_example.exe
```

#### Windows (使用MSVC)

```cmd
REM 1. 编译MuJoCo (如果还没有)
cd mujoco
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release

REM 2. 编译示例程序
cd ..
cl sparse_solver_example.c /I build\include /link build\mujoco.lib

REM 3. 运行
sparse_solver_example.exe
```

#### 使用CMake (推荐)

```bash
# 1. 创建构建目录
mkdir build_example && cd build_example

# 2. 配置
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. 编译
cmake --build .

# 4. 运行
./sparse_solver_example  # Linux/macOS
sparse_solver_example.exe  # Windows
```

## 稀疏矩阵格式

MuJoCo使用CSR (Compressed Sparse Row) 格式存储稀疏矩阵：

```c
typedef struct {
    int n;              // 矩阵维度
    int* rownnz;        // 每行的非零元素数量 [n]
    int* rowadr;        // 每行的起始地址 [n]
    int* colind;        // 列索引 [nnz]
    mjtNum* mat;        // 矩阵元素 [nnz]
} SparseMatrix;
```

### 示例：3x3稀疏矩阵

```
矩阵 A:
[ 4  1  0 ]
[ 1  5  1 ]
[ 0  1  6 ]

CSR表示:
rownnz = [2, 3, 2]      // 每行非零元素数
rowadr = [0, 2, 5]      // 每行起始地址
colind = [0, 1, 0, 1, 2, 1, 2]  // 列索引
mat    = [4, 1, 1, 5, 1, 1, 6]  // 矩阵值
```

## API参考

### 稀疏Cholesky分解

```c
// 符号分解：计算稀疏模式
int mju_cholFactorSymbolic(
    int* L_colind, int* L_rownnz, int* L_rowadr,
    int* LT_colind, int* LT_rownnz, int* LT_rowadr, int* LT_map,
    const int* rownnz, const int* rowadr, const int* colind,
    int n, mjData* d
);

// 数值分解：计算Cholesky因子
int mju_cholFactorNumeric(
    mjtNum* L, int n, mjtNum mindiag,
    const int* L_rownnz, const int* L_rowadr, const int* L_colind,
    const int* LT_rownnz, const int* LT_rowadr, const int* LT_colind,
    const int* LT_map, const mjtNum* H,
    const int* H_rownnz, const int* H_rowadr, const int* H_colind,
    mjData* d
);

// 求解：L*L'*x = b
void mju_cholSolveSparse(
    mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n,
    const int* rownnz, const int* rowadr, const int* colind
);
```

### 稀疏LU分解

```c
// LU分解
void mju_factorLUSparse(
    mjtNum *LU, int n, int* scratch,
    const int *rownnz, const int *rowadr, const int *colind, const int *index
);

// 求解
void mju_solveLUSparse(
    mjtNum *res, const mjtNum *LU, const mjtNum* vec, int n,
    const int *rownnz, const int *rowadr, const int* diag,
    const int *colind, const int *index
);
```

### 稀疏矩阵-向量乘法

```c
// res = mat * vec
void mju_mulMatVecSparse(
    mjtNum* res, const mjtNum* mat, const mjtNum* vec,
    int nr, const int* rownnz, const int* rowadr,
    const int* colind, const int* rowsuper
);

// res = mat' * vec
void mju_mulMatTVecSparse(
    mjtNum* res, const mjtNum* mat, const mjtNum* vec,
    int nr, int nc, const int* rownnz, const int* rowadr, const int* colind
);
```

### Box约束二次规划

```c
// 求解: min 0.5*x'*H*x + x'*g  s.t. lower <= x <= upper
int mju_boxQP(
    mjtNum* res, mjtNum* R, int* index,
    const mjtNum* H, const mjtNum* g, int n,
    const mjtNum* lower, const mjtNum* upper
);
```

## 性能建议

1. **矩阵选择**：
   - 对称正定矩阵：使用Cholesky分解
   - 一般方阵：使用LU分解
   - 超大规模矩阵：考虑迭代法

2. **预处理**：
   - 重新排序可以减少填充元
   - 使用符号分解预先计算稀疏模式

3. **迭代法**：
   - 适用于内存受限的情况
   - 可以使用预处理矩阵加速收敛
   - 设置合理的收敛容差

## 故障排除

### 编译错误

1. **找不到头文件**：
   ```bash
   export MUJOCO_INCLUDE_PATH=/path/to/mujoco/include
   ```

2. **找不到库文件**：
   ```bash
   export MUJOCO_LIBRARY_PATH=/path/to/mujoco/build
   ```

### 运行时错误

1. **矩阵不是正定的**：
   - Cholesky分解会失败
   - 尝试添加对角正则化：H + μI

2. **数值不稳定**：
   - 增加最小对角元素值：`mindiag`
   - 检查矩阵条件数

## 更多资源

- MuJoCo文档: https://mujoco.readthedocs.io/
- API参考: https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html
- GitHub仓库: https://github.com/google-deepmind/mujoco