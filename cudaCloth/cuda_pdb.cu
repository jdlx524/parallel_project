// cloth_simulation.cpp

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <omp.h>

// 辅助宏
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

// error 宏：用于检查 CUDA 错误
#define CUERR {                                                              \
    cudaError_t err;                                                         \
    if ((err = cudaGetLastError()) != cudaSuccess) {                         \
       std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "       \
                 << __FILE__ << ", line " << __LINE__ << std::endl;          \
       exit(1);                                                              \
    }                                                                        \
}

// HIQ 表示高精度版本（可调整网格尺寸与迭代次数）
#define HIQ

#ifndef HIQ
    #define N (64)
    #define EPS (1E-2)
    #define BIAS (0.17)
    #define ITERS (128)
    #define G (1)
#else
    #define N (128)
    #define EPS (1E-2)
    #define BIAS (0.15)
    #define ITERS (400)
    #define G (1)
#endif

// 窗口尺寸
const unsigned int window_width  = 1280;
const unsigned int window_height = 720;

// OpenGL 缓冲对象
static GLuint ibo = 0;
static GLuint vbo = 0;
static GLuint cbo = 0;
static GLuint nbo = 0;

// 主机侧数据（每 3 个浮点数一顶点，速度 3 个浮点数，颜色 4 个浮点数）
static std::vector<float> vertices;
static std::vector<float> velocities;
static std::vector<float> colors;
static std::vector<float> normals;
static std::vector<int> indices;

// 设备侧数据：通过 CUDA OpenGL 互操作获得 VBO 指针
struct cudaGraphicsResource *vbo_resource;
struct cudaGraphicsResource *vbo_resource2;  // 用于法线的 VBO

// 设备侧临时缓冲区（GPU 版 relax 操作使用）
float3 *Temp;

// 设备侧存储速度数据（按顶点，每顶点 3 个 float，共连续内存）
static float* d_velocities = nullptr;

// 设备侧锁数组，每个顶点对应一个锁（用于自碰撞时同步）
static int* d_locks = nullptr;

// 约束参数（在 init_data 中根据顶点间距计算）
static float cnstr_two;
static float cnstr_dia;

// 帧计数器
static size_t frames = 0;


/**************************************************
 * 辅助函数（设备/主机）
 **************************************************/

// 本情境下不需要地形修正，返回 0
__host__ __device__ __forceinline__
float terrain_height(float x, float y) {
    return 0.0f;
}

// update_positions：仅根据当前速度以欧拉法积分更新位置（无重力）
__host__ __device__ __forceinline__
void update_positions(float3& pos, float3& vel, const float eps) {
    pos.x += eps * vel.x;
    pos.y += eps * vel.y;
    pos.z += eps * vel.z;
}

// adjust_positions：本情境下不做额外校正（保留空操作）
__host__ __device__ __forceinline__
void adjust_positions(float3 &pos) {
    // no-op
}

// relax_constraint：用于约束相邻顶点距离
__host__ __device__ __forceinline__
void relax_constraint(const float3 *Pos, float3 *Tmp,
                      const int l, const int m,
                      const float constraint, const float bias) {
    float3 delta;
    delta.x = Pos[l].x - Pos[m].x;
    delta.y = Pos[l].y - Pos[m].y;
    delta.z = Pos[l].z - Pos[m].z;
    const float invlen = rsqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    const float factor = (1.0f - constraint * invlen) * bias;
#if defined(__CUDA_ARCH__)
    atomicAdd(&Tmp[l].x, -delta.x * factor);
    atomicAdd(&Tmp[l].y, -delta.y * factor);
    atomicAdd(&Tmp[l].z, -delta.z * factor);
    atomicAdd(&Tmp[m].x, +delta.x * factor);
    atomicAdd(&Tmp[m].y, +delta.y * factor);
    atomicAdd(&Tmp[m].z, +delta.z * factor);
#else
    Tmp[l].x -= delta.x * factor;
    Tmp[l].y -= delta.y * factor;
    Tmp[l].z -= delta.z * factor;
    Tmp[m].x += delta.x * factor;
    Tmp[m].y += delta.y * factor;
    Tmp[m].z += delta.z * factor;
#endif
}

// normalize：将向量归一化
__host__ __device__ __forceinline__
void normalize(float3 &normal) {
    const float invrho = rsqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    normal.x *= invrho;
    normal.y *= invrho;
    normal.z *= invrho;
}

// wedge：计算局部面片法线贡献
__host__ __device__ __forceinline__
void wedge(const float3 *Vertices, float3 &Normal,
           const int &i, const int &j, const int n,
           const int &a, const int &b) {
    float3 center = Vertices[i*n+j];
    float3 span_u = Vertices[(i+a)*n+j];
    float3 span_v = Vertices[i*n+(j+b)];
    span_u.x -= center.x; span_u.y -= center.y; span_u.z -= center.z;
    span_v.x -= center.x; span_v.y -= center.y; span_v.z -= center.z;
    float3 cross;
    cross.x = span_u.y * span_v.z - span_v.y * span_u.z;
    cross.y = span_u.z * span_v.x - span_v.z * span_u.x;
    cross.z = span_u.x * span_v.y - span_v.x * span_u.y;
    Normal.x += cross.x * a * b;
    Normal.y += cross.y * a * b;
    Normal.z += cross.z * a * b;
}

/**************************************************
 * CUDA 内核函数
 **************************************************/

// 传播内核：根据速度更新位置
__global__
void propagate_kernel(float3 *Vertices, float3 *Velocities, const int n, const float eps=EPS) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n)
        update_positions(Vertices[i*n+j], Velocities[i*n+j], eps);
}

// 校正内核：根据约束迭代调整位置（调用 relax_constraint）
__global__
void validate_kernel(float3 *Vertices, float3 *Temp,
                     const float cnstr_two, const float cnstr_dia,
                     const int n, const float bias=BIAS) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n-1 && j < n)
        relax_constraint(Vertices, Temp, i*n+j, (i+1)*n+j, cnstr_two, bias);
    if(i < n && j < n-1)
        relax_constraint(Vertices, Temp, i*n+j, i*n+(j+1), cnstr_two, bias);
    if(i < n-2 && j < n)
        relax_constraint(Vertices, Temp, i*n+j, (i+2)*n+j, 2*cnstr_two, bias);
    if(i < n && j < n-2)
        relax_constraint(Vertices, Temp, i*n+j, i*n+(j+2), 2*cnstr_two, bias);
    if(i < n-1 && j < n-1)
        relax_constraint(Vertices, Temp, i*n+j, (i+1)*n+(j+1), cnstr_dia, bias);
    if(i > 0 && i < n && j < n-1)
        relax_constraint(Vertices, Temp, i*n+j, (i-1)*n+(j+1), cnstr_dia, bias);
}

// 调整内核：调用 adjust_positions（目前为空操作）
__global__
void adjust_kernel(float3* Temp, const int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n)
        adjust_positions(Temp[i*n+j]);
}

// 更新法线内核：根据顶点数据计算法线
__global__
void update_normals_kernel(float3 *Vertices, float3 *Normals, const int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n) {
        float3 Normal = {0, 0, 0};
        if(i > 0 && j > 0)
            wedge(Vertices, Normal, i, j, n, -1, -1);
        if(i > 0 && j+1 < n)
            wedge(Vertices, Normal, i, j, n, -1, +1);
        if(i+1 < n && j > 0)
            wedge(Vertices, Normal, i, j, n, +1, -1);
        if(i+1 < n && j+1 < n)
            wedge(Vertices, Normal, i, j, n, +1, +1);
        normalize(Normal);
        Normals[i*n+j] = Normal;
    }
}

// 自碰撞内核（CUDA版）
// 每个线程针对一个顶点 i，遍历 j > i，检测碰撞，若碰撞则更新位置及消除相对速度
__global__
void self_collision_kernel(float* vertices, float* velocities,
                           int n, float collisionThreshold, int total, int* locks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int row_i = i / n, col_i = i % n;

    for (int j = i + 1; j < total; j++) {
        int row_j = j / n, col_j = j % n;
        // 排除局部邻域（相邻点不考虑）
        if (abs(row_i - row_j) <= 1 && abs(col_i - col_j) <= 1)
            continue;

        // 读取顶点位置（每个顶点 3 个 float）
        float xi = vertices[i*3];
        float yi = vertices[i*3 + 1];
        float zi = vertices[i*3 + 2];
        float xj = vertices[j*3];
        float yj = vertices[j*3 + 1];
        float zj = vertices[j*3 + 2];

        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float dist2 = dx*dx + dy*dy + dz*dz;

        if(dist2 < collisionThreshold * collisionThreshold) {
            // 计算平均位置
            float avgx = 0.5f * (xi + xj);
            float avgy = 0.5f * (yi + yj);
            float avgz = 0.5f * (zi + zj);

            // 采用锁机制：先锁下标较小的，再锁较大的
            while (atomicCAS(&locks[i], 0, 1) != 0);
            while (atomicCAS(&locks[j], 0, 1) != 0);

            // 更新两个顶点位置（写回平均值）
            vertices[i*3]     = avgx;
            vertices[i*3 + 1] = avgy;
            vertices[i*3 + 2] = avgz;
            vertices[j*3]     = avgx;
            vertices[j*3 + 1] = avgy;
            vertices[j*3 + 2] = avgz;

            // 消除相对速度在碰撞方向上的分量
            float vix = velocities[i*3];
            float viy = velocities[i*3 + 1];
            float viz = velocities[i*3 + 2];
            float vjx = velocities[j*3];
            float vjy = velocities[j*3 + 1];
            float vjz = velocities[j*3 + 2];
            float rvx = vix - vjx;
            float rvy = viy - vjy;
            float rvz = viz - vjz;
            float norm = sqrtf(dx*dx + dy*dy + dz*dz);
            if(norm > 0){
                float nx = dx / norm;
                float ny = dy / norm;
                float nz = dz / norm;
                float vrel = rvx * nx + rvy * ny + rvz * nz;
                float correction = vrel / 2.0f;
                velocities[i*3]     -= correction * nx;
                velocities[i*3 + 1] -= correction * ny;
                velocities[i*3 + 2] -= correction * nz;
                velocities[j*3]     += correction * nx;
                velocities[j*3 + 1] += correction * ny;
                velocities[j*3 + 2] += correction * nz;
            }

            // 释放锁
            atomicExch(&locks[i], 0);
            atomicExch(&locks[j], 0);
        }
    }
}

/**************************************************
 * GPU 包装调用函数
 **************************************************/
 __global__
void copy_kernel(float3 *target, float3 *source, int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n)
        target[i*n+j] = source[i*n+j];
}

// propagate_gpu：利用 CUDA 内核更新顶点位置（欧拉积分）
void propagate_gpu(int n) {
    float3 *d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vertices, &num_bytes, vbo_resource);

    dim3 grid((n+7)/8, (n+7)/8, 1), blck(8,8,1);
    // 利用 reinterpret_cast 将 d_velocities 转换为 float3*（注意保证内存连续性，每 3 个 float 为一个 float3）
    propagate_kernel<<<grid, blck>>>(d_vertices, reinterpret_cast<float3*>(d_velocities), n);
    CUERR
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

// validate_gpu：利用 CUDA 内核多次迭代校正顶点间距离
void validate_gpu(int n, const int iters=ITERS) {
    float3 *d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vertices, &num_bytes, vbo_resource);

    dim3 grid((n+7)/8, (n+7)/8, 1), blck(8,8,1);
    for(int iter = 0; iter < iters; iter++){
        // 将当前顶点数据复制到临时缓冲区 Temp
        copy_kernel<<<grid, blck>>>(Temp, d_vertices, n);
        CUERR
        validate_kernel<<<grid, blck>>>(d_vertices, Temp, cnstr_two, cnstr_dia, n);
        CUERR
        adjust_kernel<<<grid, blck>>>(Temp, n);
        CUERR
        copy_kernel<<<grid, blck>>>(d_vertices, Temp, n);
        CUERR
    }
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

// 更新法线 GPU 版本
void update_normals_gpu(int n) {
    float3 *d_vertices, *d_normals;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vertices, &num_bytes, vbo_resource);
    cudaGraphicsMapResources(1, &vbo_resource2, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_normals, &num_bytes, vbo_resource2);

    dim3 grid((n+7)/8, (n+7)/8, 1), blck(8,8,1);
    update_normals_kernel<<<grid, blck>>>(d_vertices, d_normals, n);
    CUERR
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
    cudaGraphicsUnmapResources(1, &vbo_resource2, 0);
}

// 自碰撞 GPU 版本包装函数
void self_collision_gpu(int n) {
    float* d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vertices, &num_bytes, vbo_resource);

    int total = n * n;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    self_collision_kernel<<<numBlocks, blockSize>>>(d_vertices, d_velocities, n, 0.5f * cnstr_two, total, d_locks);
    cudaDeviceSynchronize();
    CUERR
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}


/**************************************************
 * CPU 版本辅助函数（供对比，非 GPU 仿真时使用）
 **************************************************/

void propagate(float3 *vertices, float3 *velocities, const int n, const float eps=EPS) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            update_positions(vertices[i*n+j], velocities[i*n+j], eps);
        }
    }
}

void validate(float3 *vertices, const int n, const int iters=ITERS, const float bias=BIAS) {
    std::vector<float3> temp(3*n*n);
    for(int iter = 0; iter < iters; iter++){
        std::memcpy(temp.data(), vertices, sizeof(float3)*n*n);
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < n-1; i++){
            for(int j = 0; j < n; j++){
                relax_constraint(vertices, temp.data(), i*n+j, (i+1)*n+j, cnstr_two, bias);
            }
        }
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n-1; j++){
                relax_constraint(vertices, temp.data(), i*n+j, i*n+(j+1), cnstr_two, bias);
            }
        }
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < n-2; i++){
            for(int j = 0; j < n; j++){
                relax_constraint(vertices, temp.data(), i*n+j, (i+2)*n+j, 2*cnstr_two, bias);
            }
        }
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n-2; j++){
                relax_constraint(vertices, temp.data(), i*n+j, i*n+(j+2), 2*cnstr_two, bias);
            }
        }
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < n-1; i++){
            for(int j = 0; j < n-1; j++){
                relax_constraint(vertices, temp.data(), i*n+j, (i+1)*n+(j+1), cnstr_dia, bias);
            }
        }
        #pragma omp parallel for collapse(2)
        for(int i = 1; i < n; i++){
            for(int j = 0; j < n-1; j++){
                relax_constraint(vertices, temp.data(), i*n+j, (i-1)*n+(j+1), cnstr_dia, bias);
            }
        }
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                adjust_positions(temp[i*n+j]);
            }
        }
        std::memcpy(vertices, temp.data(), sizeof(float3)*n*n);
    }
}

void update_normals(float3 *vertices, float3 *normals, const int n) {
    std::memset(normals, 0, sizeof(float3)*n*n);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float3 normal = {0, 0, 0};
            if(i > 0 && j > 0)
                wedge(vertices, normal, i, j, n, -1, -1);
            if(i > 0 && j+1 < n)
                wedge(vertices, normal, i, j, n, -1, +1);
            if(i+1 < n && j > 0)
                wedge(vertices, normal, i, j, n, +1, -1);
            if(i+1 < n && j+1 < n)
                wedge(vertices, normal, i, j, n, +1, +1);
            normalize(normal);
            normals[i*n+j] = normal;
        }
    }
}


/**************************************************
 * OpenGL 渲染函数：完整 display 函数（采用全部 GPU 版仿真）
 **************************************************/
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 使布料整体缓慢旋转，便于多角度观察
    // glRotated((frames++) * 0.2, 0, 0, 1);

    // GPU 版仿真更新：传播、约束校正、自碰撞、更新法线
    propagate_gpu(N);
    validate_gpu(N, ITERS);
    self_collision_gpu(N);
    update_normals_gpu(N);

    // 启用顶点、颜色、法线、索引数组的客户端状态
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);

    // 绑定顶点数据（VBO 已被 GPU 内核更新）
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, NULL);

    // 绑定颜色数据
    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glColorPointer(4, GL_FLOAT, 0, NULL);

    // 绑定法线数据
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glNormalPointer(GL_FLOAT, 0, NULL);

    // 绘制布料网格（采用三角带）
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLE_STRIP, indices.size(), GL_UNSIGNED_INT, NULL);

    // 关闭客户端状态
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    // 每隔一定帧数重新初始化布料数据，重置仿真状态
    if((int)(frames * G) % 1000 == 0)
        ; // 可根据需要调用 init_data(N) 重新初始化

    glutSwapBuffers();
    glutPostRedisplay();
}


/**************************************************
 * init_data: 初始化布料数据，并分配相关设备内存
 * 这里构造主视倒 V 形态，左半边（y 较小）运动速度较大
 **************************************************/
void init_data(int n) {
    // 布料在 x,y 坐标上的取值范围：[-2, +2]
    float offsetX = 0.0f;
    float offsetY = 0.0f;

    // 设置形态参数：baseline 为中间最高，amplitude 控制下降幅度
    float baseline = 0.8f;
    float amplitude = 0.3f;

    float min_x = -2.0f + offsetX;
    float max_x =  2.0f + offsetX;
    float min_y = -2.0f + offsetY;
    float max_y =  2.0f + offsetY;
    // 为主视效果，以 y 坐标构造倒 V 形态，以 y 中心作为最高处
    float center_y = (min_y + max_y) / 2.0f;

    vertices.clear();
    velocities.clear();
    std::srand((unsigned)std::time(nullptr));

    // 生成网格：i 控制 x 坐标，j 控制 y 坐标
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float x = i * 4.0f/(n-1) - 2.0f + offsetX;
            float y = j * 4.0f/(n-1) - 2.0f + offsetY;
            // 生成噪声 ∈ [-0.05, +0.05]
            float noise = (((std::rand() % 1000) / 1000.0f) - 0.5f) * 0.1f;
            // 利用 y 坐标构造倒 V：中间（y=center_y）最高，两侧降低
            float z = baseline - amplitude * fabs(y - center_y) + noise;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // 初始速度：采用 y 方向运动
            // 当 y < center_y 时（左侧）速度为正，且较大；y > center_y 时速度为负，较小；正中为 0
            float v_left  = 1.0f;  // 左侧最大速度
            float v_right = 0.5f;  // 右侧最大速度
            float vy;
            if (y < center_y) {
                vy = v_left * ((center_y - y) / (center_y - min_y));
            } else if (y > center_y) {
                vy = - v_right * ((y - center_y) / (max_y - center_y));
            } else {
                vy = 0.0f;
            }
            float vx = 0.0f;
            float vz = 0.0f;
            velocities.push_back(vx);
            velocities.push_back(vy);
            velocities.push_back(vz);
        }
    }

    // 分配 GPU 版 relax 操作的临时缓冲区 Temp，并置 0
    cudaMalloc(&Temp, sizeof(float3)*n*n); CUERR
    cudaMemset(Temp, 0, sizeof(float3)*n*n); CUERR

    // 根据布料网格计算约束参数
    // 这里以 x 坐标上相邻顶点距离为参考（均为 4/(n-1)）
    cnstr_two = vertices[3*n] - vertices[0];
    cnstr_dia = sqrt(2 * cnstr_two * cnstr_two);

    // 初始化法线数据
    normals.resize(3*n*n, 0.0f);
    update_normals((float3*) vertices.data(), (float3*) normals.data(), n);

    // 初始化颜色数据：全部设置为红色（RGBA:1,0,0,0.9）
    colors.clear();
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            colors.push_back(1.0f); // R
            colors.push_back(0.0f); // G
            colors.push_back(0.0f); // B
            colors.push_back(0.9f); // A
        }
    }

    // 构造索引数组，用于三角带绘制
    indices.clear();
    for (int i = 0; i < n-1; i++){
        int base = i * n;
        indices.push_back(base);
        for (int j = 0; j < n; j++){
            indices.push_back(base + j);
            indices.push_back(base + j + n);
        }
        indices.push_back(base + 2*n - 1);
    }

    // 创建并上传 OpenGL 缓冲区数据
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &cbo);
    glGenBuffers(1, &nbo);
    glGenBuffers(1, &ibo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*normals.size(), normals.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&vbo_resource2, nbo, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*colors.size(), colors.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*indices.size(), indices.data(), GL_STATIC_DRAW);

    // 分配设备侧速度数组 d_velocities，并将 host 侧数据复制过去
    int total = n * n;
    if (d_velocities) cudaFree(d_velocities);
    cudaMalloc((void**)&d_velocities, total * 3 * sizeof(float));
    cudaMemcpy(d_velocities, velocities.data(), total * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // 分配设备侧锁数组 d_locks，初值置 0
    if (d_locks) cudaFree(d_locks);
    cudaMalloc((void**)&d_locks, total * sizeof(int));
    cudaMemset(d_locks, 0, total * sizeof(int));
}

/**************************************************
 * OpenGL 初始化函数
 **************************************************/
void init_GL(int *argc, char **argv) {
    std::srand((unsigned)std::time(nullptr));
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cloth Simulation: Main-View Inverted V with Left>Right Velocity (CUDA)");

    glutDisplayFunc(display);

    glewInit();

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLfloat mat_specular[] = {0.8f, 0.8f, 0.8f, 1.0f};
    GLfloat mat_shininess[] = {50.0f};
    GLfloat light_position[] = {1.0f, 1.0f, 1.0f, 0.0f};
    glShadeModel(GL_SMOOTH);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width/(GLfloat)window_height, 0.1, 10.0);
    glTranslatef(0.0, 0.0, -6.0);
    // 调整视角，使主视效果呈现倒 V 状态
    glRotated(300, 1, 0, 0);
    glRotated(270, 0, 0, 1);
}

/**************************************************
 * 主函数
 **************************************************/
int main(int argc, char **argv) {
    init_GL(&argc, argv);
    init_data(N);
    glutMainLoop();
    return 0;
}
