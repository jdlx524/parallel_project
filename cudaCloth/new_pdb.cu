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

// 主机侧数据
static std::vector<float> vertices;    // 每 3 个浮点数一顶点 (x,y,z)
static std::vector<float> velocities;  // 每 3 个浮点数一速度
static std::vector<float> colors;      // 每 4 个浮点数一颜色 (rgba)
static std::vector<float> normals;     // 每 3 个浮点数一法线
static std::vector<int> indices;       // 用于绘制三角带

// 设备侧数据（用于 CUDA 映射 OpenGL VBO 等）
struct cudaGraphicsResource *vbo_resource;
struct cudaGraphicsResource *vbo_resource2;
void *g_vbo_buffer;
void *g_vbo_buffer2;
float3 * Velocities;
float3 * Temp;

// 约束参数（在 init_data 中根据顶点间距计算）
static float cnstr_two;
static float cnstr_dia;

// 帧计数器
static size_t frames = 0;

/**************************************************
 * 函数声明
 **************************************************/
void display();
void init_GL(int *argc, char **argv);
void init_data(int);
void self_collision_cpu(std::vector<float>& vertices, std::vector<float>& velocities, int n);

/**************************************************
 * 辅助函数（设备/主机）
 **************************************************/

// 本情境下不需要地形修正，返回 0（仅用于兼容 relax_constraint 算法）
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
 * （为兼容 GPU 版本，这里保留与 CPU 版本类似的内核实现）
 **************************************************/

__global__
void copy_kernel(float3 *target, float3 *source, int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n)
        target[i*n+j] = source[i*n+j];
}

__global__
void propagate_kernel(float3 *Vertices, float3 *Velocities, const int n, const float eps=EPS) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n)
        update_positions(Vertices[i*n+j], Velocities[i*n+j], eps);
}

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

__global__
void adjust_kernel(float3* Temp, const int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n)
        adjust_positions(Temp[i*n+j]);
}

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

/**************************************************
 * GPU 包装调用函数（可选）
 **************************************************/
void propagate_gpu(int n) {
    float3 *Vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Vertices, &num_bytes, vbo_resource);
    dim3 grid((n+7)/8, (n+7)/8, 1), blck(8,8,1);
    propagate_kernel<<<grid, blck>>>(Vertices, Velocities, n); CUERR
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

void validate_gpu(int n, const int iters=ITERS) {
    float3 *Vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Vertices, &num_bytes, vbo_resource);
    dim3 grid((n+7)/8, (n+7)/8, 1), blck(8,8,1);
    for(int iter = 0; iter < iters; iter++){
        copy_kernel<<<grid, blck>>>(Temp, Vertices, n);
        validate_kernel<<<grid, blck>>>(Vertices, Temp, cnstr_two, cnstr_dia, n);
        adjust_kernel<<<grid, blck>>>(Temp, n);
        copy_kernel<<<grid, blck>>>(Vertices, Temp, n);
    }
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

void update_normals_gpu(int n) {
    float3 *Vertices, *Normals;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Vertices, &num_bytes, vbo_resource);
    cudaGraphicsMapResources(1, &vbo_resource2, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Normals, &num_bytes, vbo_resource2);
    dim3 grid((n+7)/8, (n+7)/8, 1), blck(8,8,1);
    update_normals_kernel<<<grid, blck>>>(Vertices, Normals, n); CUERR
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
    cudaGraphicsUnmapResources(1, &vbo_resource2, 0);
}

/**************************************************
 * CPU 版本的仿真函数
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
 * 布料自碰撞（CPU版）——相对速度清零
 * 简单版本：遍历所有非局部邻域顶点，若距离小于阈值，则粘合并消除沿碰撞方向相对速度
 **************************************************/
void self_collision_cpu(std::vector<float>& vertices, std::vector<float>& velocities, int n) {
    int total = n * n;
    float collisionThreshold = 0.5f * cnstr_two;
    for (int i = 0; i < total; i++) {
        int idx_i = i * 3;
        for (int j = i + 1; j < total; j++) {
            int idx_j = j * 3;
            // 排除局部邻域（相邻点不考虑）
            int row_i = i / n, col_i = i % n;
            int row_j = j / n, col_j = j % n;
            if(std::abs(row_i - row_j) <= 1 && std::abs(col_i - col_j) <= 1)
                continue;
            float dx = vertices[idx_i]     - vertices[idx_j];
            float dy = vertices[idx_i + 1] - vertices[idx_j + 1];
            float dz = vertices[idx_i + 2] - vertices[idx_j + 2];
            float dist2 = dx*dx + dy*dy + dz*dz;
            if(dist2 < collisionThreshold * collisionThreshold) {
                // 取平均位置
                float avgx = 0.5f * (vertices[idx_i] + vertices[idx_j]);
                float avgy = 0.5f * (vertices[idx_i + 1] + vertices[idx_j + 1]);
                float avgz = 0.5f * (vertices[idx_i + 2] + vertices[idx_j + 2]);
                vertices[idx_i] = vertices[idx_j] = avgx;
                vertices[idx_i+1] = vertices[idx_j+1] = avgy;
                vertices[idx_i+2] = vertices[idx_j+2] = avgz;
                // 消除相对速度在碰撞方向分量
                float vix = velocities[idx_i];
                float viy = velocities[idx_i+1];
                float viz = velocities[idx_i+2];
                float vjx = velocities[idx_j];
                float vjy = velocities[idx_j+1];
                float vjz = velocities[idx_j+2];
                float rvx = vix - vjx;
                float rvy = viy - vjy;
                float rvz = viz - vjz;
                float norm = std::sqrt(dx*dx + dy*dy + dz*dz);
                if(norm > 0){
                    float nx = dx / norm;
                    float ny = dy / norm;
                    float nz = dz / norm;
                    float vrel = rvx * nx + rvy * ny + rvz * nz;
                    float correction = vrel / 2.0f;
                    velocities[idx_i]     -= correction * nx;
                    velocities[idx_i+1]   -= correction * ny;
                    velocities[idx_i+2]   -= correction * nz;
                    velocities[idx_j]     += correction * nx;
                    velocities[idx_j+1]   += correction * ny;
                    velocities[idx_j+2]   += correction * nz;
                }
            }
        }
    }
}

/**************************************************
 * OpenGL 渲染函数
 **************************************************/
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 这里通过旋转使布料整体缓慢旋转，便于多角度观察
    glRotated((frames++) * 0.2, 0, 0, 1);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);

    // 采用 CPU 模型更新
    propagate((float3 *) vertices.data(), (float3*) velocities.data(), N);
    validate((float3 *) vertices.data(), N);
    self_collision_cpu(vertices, velocities, N);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STREAM_DRAW);
    glVertexPointer(3, GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glColorPointer(4, GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    update_normals((float3*) vertices.data(), (float3*) normals.data(), N);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*normals.size(), normals.data(), GL_STREAM_DRAW);
    glNormalPointer(GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLE_STRIP, indices.size(), GL_UNSIGNED_INT, NULL);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    if((int)(frames * G) % 1000 == 0)
        init_data(N);

    glutSwapBuffers();
    glutPostRedisplay();
}

/**************************************************
 * init_data: 初始化布料数据，生成拱形(∩)形态，左边初速度大于右边
 **************************************************/
 void init_data(int n) {
    // 坐标偏移（这里暂不偏移）
    float offsetX = 0.0f;
    float offsetY = 0.0f;

    // 设置布料初始形态参数
    // baseline 为中间最高（倒 V 型的“顶”），amplitude 控制两侧相对降低的幅度
    float baseline = 0.8f;    // 中心高度
    float amplitude = 0.3f;   // 倒 V 的坡度参数

    // 原先 x 与 y 坐标均在 [-2,+2] 范围内，这里我们继续保持这种网格分布
    float min_x = -2.0f + offsetX;
    float max_x =  2.0f + offsetX;
    float min_y = -2.0f + offsetY;
    float max_y =  2.0f + offsetY;
    // 为主视效果，我们以 y 坐标构造倒 V 形态，因此选取 y 的中间值作为对称中心
    float center_y = (min_y + max_y) / 2.0f;  // 一般为 0

    vertices.clear();
    velocities.clear();
    std::srand((unsigned)std::time(nullptr));

    // 生成网格数据：i 方向控制 x 坐标（保持平铺），j 方向控制 y 坐标
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // 布料网格中每个点的 (x, y) 坐标
            float x = i * 4.0f/(n-1) - 2.0f + offsetX;
            float y = j * 4.0f/(n-1) - 2.0f + offsetY;
            // 加入一定随机噪声，范围在 [-0.05, +0.05]
            float noise = (((std::rand() % 1000) / 1000.0f) - 0.5f) * 0.1f;
            // 修改：利用 y 坐标构造倒 V 形态
            // 计算方式：中间（y = center_y）处最高 (z = baseline)，两侧按离中心距离降低
            float z = baseline - amplitude * fabs(y - center_y) + noise;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // 设定初始速度：为了使两侧向中间运动，
            // 我们令位于左侧（原 y 较小，即 y < center_y）的点速度为正（向右），而右侧点（y > center_y）的速度为负（向左）。
            // 同时左侧的绝对速度设为较大（例如 1.0），右侧较小（例如 0.5），
            // 这里我们采用线性插值使得最左（y = min_y）取最大速度，最右（y = max_y）取最小速度，
            // 中间（y = center_y）处速度为 0，从而使两侧在碰撞时发生压缩。
            float v_left = 1.0f;   // 左侧（屏幕左，原 y 较小）的最大速度
            float v_right = 0.5f;  // 右侧（屏幕右，原 y 较大）的最大速度
            float vy;
            if (y < center_y) {
                // 当 y 处于左半边时，做线性内插：y 从 min_y 到 center_y，速度从 v_left 到 0
                vy = v_left * ((center_y - y) / (center_y - min_y));
            } else if (y > center_y) {
                // 右半边：y 从 center_y 到 max_y，速度从 0 到 -v_right
                vy = - v_right * ((y - center_y) / (max_y - center_y));
            } else {
                vy = 0.0f;
            }
            // 为简单起见，令 x 和 z 方向的初始速度为 0
            float vx = 0.0f;
            float vz = 0.0f;
            velocities.push_back(vx);
            velocities.push_back(vy);
            velocities.push_back(vz);
        }
    }

    // 分配 Device 侧用于 relax 操作的临时缓冲区 Temp，并置 0
    cudaMalloc(&Temp, sizeof(float3)*n*n); CUERR
    cudaMemset(Temp, 0, sizeof(float3)*n*n); CUERR

    // 约束参数：这里仍以网格内相邻顶点间的初始距离为参考
    // 此时约束距离根据 x 坐标（或 y 坐标，均为 4/(n-1)）计算得到
    cnstr_two = vertices[3*n] - vertices[0];
    cnstr_dia = sqrt(2 * cnstr_two * cnstr_two);

    // 初始化法线数据
    normals.resize(3*n*n, 0.0f);
    update_normals((float3*) vertices.data(), (float3*) normals.data(), n);

    // 初始化颜色：设置为红色，透明度 0.9
    colors.clear();
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            colors.push_back(1.0f); // R
            colors.push_back(0.0f); // G
            colors.push_back(0.0f); // B
            colors.push_back(0.9f); // A
        }
    }

    // 构造索引数组，用于三角带绘制布料网格
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
}


/**************************************************
 * OpenGL 初始化函数
 **************************************************/
void init_GL(int *argc, char **argv) {
    std::srand((unsigned)std::time(nullptr));
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cloth Simulation: Inverted V View with Left>Right Velocity");
    glutDisplayFunc(display);

    glewInit();

    glClearColor(1.0,1.0,1.0,1.0);
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
    glTranslatef(0.0,0.0,-6.0);
    // 此处调整旋转角度，使主视图呈现倒 V 效果
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
