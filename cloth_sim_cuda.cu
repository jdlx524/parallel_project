// cloth_sim_cuda_explicit.cu
// 注：本代码使用 CUDA 进行物理计算，但不使用 CUDA-OpenGL 互操作，而是通过显式的 cudaMemcpy 从设备复制数据到主机，再由 OpenGL 更新 VBO。
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

// CUDA 内置 float3 类型可以直接使用

// 根据 HIQ 宏选择测试规模和迭代次数
#define HIQ
#ifdef HIQ
    #define N (128)
    #define EPS (1E-2)
    #define BIAS (0.15f)
    #define ITERS (400)
    #define G (1)
#else
    #define N (64)
    #define EPS (1E-2)
    #define BIAS (0.17f)
    #define ITERS (128)
    #define G (1)
#endif

/**************************************************
 * 常量设置
 **************************************************/
const unsigned int window_width = 1920;
const unsigned int window_height = 1080;

// OpenGL 缓冲对象
static GLuint ibo = 0;
static GLuint vbo = 0;
static GLuint cbo = 0;
static GLuint nbo = 0;

// host 侧数据：顶点、颜色、法线、indices（连续数组）  
// 每个顶点由3个 float 组成，颜色为 RGBA（4个 float）
static std::vector<float> vertices;
static std::vector<float> colors;
static std::vector<float> normals;
static std::vector<int> indices;

// 设备侧数据指针
static float3* d_vertices = nullptr;
static float3* d_velocities = nullptr;
static float3* d_normals = nullptr;

// 约束参数，全局变量
static float cnstr_two;
static float cnstr_dia;

// 帧计数
static size_t frames = 0;

/**************************************************
 * 设备侧辅助函数（__host__ __device__）
 **************************************************/

// 更新顶点位置：若点的距离平方 > 1，则在 z 方向上施加修正；然后按速度积分更新位置
__host__ __device__ __forceinline__
void update_positions_device(float3 &pos, float3 &vel, const float eps) {
    float dist2 = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    if(dist2 > 1.0f)
        vel.z -= eps * G;
    pos.x += eps * vel.x;
    pos.y += eps * vel.y;
    pos.z += eps * vel.z;
}

// 将点"贴"到球面上：若距离大于 1，则按比例缩放到 1
__host__ __device__ __forceinline__
void adjust_positions_device(float3 &pos) {
    float len2 = pos.x*pos.x + pos.y*pos.y + pos.z*pos.z;
    float invrho = rsqrtf(len2);
    pos.x *= (invrho < 1.0f ? invrho : 1.0f);
    pos.y *= (invrho < 1.0f ? invrho : 1.0f);
    pos.z *= (invrho < 1.0f ? invrho : 1.0f);
}

// 对两个点进行约束修正，使它们的距离趋向于目标 distance（constraint）
__host__ __device__ __forceinline__
void relax_constraint_device(const float3* Pos, float3* Tmp,
                             const int l, const int m,
                             const float constraint, const float bias) {
    float3 delta;
    delta.x = Pos[l].x - Pos[m].x;
    delta.y = Pos[l].y - Pos[m].y;
    delta.z = Pos[l].z - Pos[m].z;
    float len2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
    float invlen = rsqrtf(len2);
    float factor = (1.0f - constraint * invlen) * bias;
    Tmp[l].x -= delta.x * factor;
    Tmp[l].y -= delta.y * factor;
    Tmp[l].z -= delta.z * factor;
    Tmp[m].x += delta.x * factor;
    Tmp[m].y += delta.y * factor;
    Tmp[m].z += delta.z * factor;
}

// 计算交叉乘并累加到 Normal
__host__ __device__ __forceinline__
void wedge_device(const float3* Vertices, float3 &Normal,
                  const int i, const int j, const int n,
                  const int a, const int b) {
    int idx = i * n + j;
    float3 center = Vertices[idx];
    float3 span_u = Vertices[(i+a) * n + j];
    float3 span_v = Vertices[i * n + (j+b)];
    
    span_u.x -= center.x;
    span_u.y -= center.y;
    span_u.z -= center.z;
    
    span_v.x -= center.x;
    span_v.y -= center.y;
    span_v.z -= center.z;
    
    float3 cross;
    cross.x = span_u.y * span_v.z - span_v.y * span_u.z;
    cross.y = span_u.z * span_v.x - span_v.z * span_u.x;
    cross.z = span_u.x * span_v.y - span_v.x * span_u.y;
    
    Normal.x += cross.x * a * b;
    Normal.y += cross.y * a * b;
    Normal.z += cross.z * a * b;
}

__host__ __device__ __forceinline__
void normalize_device(float3 &normal) {
    float len2 = normal.x*normal.x + normal.y*normal.y + normal.z*normal.z;
    float invrho = rsqrtf(len2);
    normal.x *= invrho;
    normal.y *= invrho;
    normal.z *= invrho;
}

/**************************************************
 * CUDA 核函数（GPU 部分）
 **************************************************/

// propagate_kernel：更新每个顶点的位置（积分）
__global__
void propagate_kernel(float3* vertices, float3* velocities, const int n, const float eps) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n) {
        int idx = i * n + j;
        update_positions_device(vertices[idx], velocities[idx], eps);
    }
}

// validate_kernel：对邻域约束进行修正（这里只处理水平、垂直和一条对角约束）
__global__
void validate_kernel(float3* vertices, float3* temp,
                     const float cnstr_two, const float cnstr_dia,
                     const int n, const float bias) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n) {
        int idx = i * n + j;
        if(i < n - 1)
            relax_constraint_device(vertices, temp, idx, idx + n, cnstr_two, bias);
        if(j < n - 1)
            relax_constraint_device(vertices, temp, idx, idx + 1, cnstr_two, bias);
        if(i < n - 1 && j < n - 1)
            relax_constraint_device(vertices, temp, idx, idx + n + 1, cnstr_dia, bias);
    }
}

// adjust_kernel：将 temp 中的数据投影到球面上
__global__
void adjust_kernel(float3* temp, const int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n) {
        int idx = i * n + j;
        adjust_positions_device(temp[idx]);
    }
}

// update_normals_kernel：利用相邻 4 个顶点计算法线
__global__
void update_normals_kernel(const float3* vertices, float3* normals, const int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < n) {
        int idx = i * n + j;
        float3 normal = {0.0f, 0.0f, 0.0f};
        if(i > 0 && j > 0)
            wedge_device(vertices, normal, i, j, n, -1, -1);
        if(i > 0 && j < n - 1)
            wedge_device(vertices, normal, i, j, n, -1, +1);
        if(i < n - 1 && j > 0)
            wedge_device(vertices, normal, i, j, n, +1, -1);
        if(i < n - 1 && j < n - 1)
            wedge_device(vertices, normal, i, j, n, +1, +1);
        normalize_device(normal);
        normals[idx] = normal;
    }
}

/**************************************************
 * CUDA 物理计算调用（主机侧）
 **************************************************/

// simulate_gpu：在设备端 d_vertices, d_velocities, d_normals 上执行物理更新
void simulate_gpu(const int n, const float eps, const int iters, const float bias) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1)/block.x, (n + block.y - 1)/block.y);
    
    // 调用 propagate_kernel 更新位置
    propagate_kernel<<<grid, block>>>(d_vertices, d_velocities, n, eps);
    
    // 申请一个临时设备缓冲区 d_temp 用于约束迭代
    float3* d_temp = nullptr;
    cudaMalloc(&d_temp, sizeof(float3) * n * n);
    
    for (int iter = 0; iter < iters; iter++) {
        cudaMemcpy(d_temp, d_vertices, sizeof(float3) * n * n, cudaMemcpyDeviceToDevice);
        validate_kernel<<<grid, block>>>(d_vertices, d_temp, cnstr_two, cnstr_dia, n, bias);
        adjust_kernel<<<grid, block>>>(d_temp, n);
        cudaMemcpy(d_vertices, d_temp, sizeof(float3) * n * n, cudaMemcpyDeviceToDevice);
    }
    cudaFree(d_temp);
    
    // 更新法线
    update_normals_kernel<<<grid, block>>>(d_vertices, d_normals, n);
}

/**************************************************
 * OpenGL 渲染与显示
 **************************************************/
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotated((frames++) * 0.2, 0, 0, 1);
    
    if (frames % 500 == 0)
        std::cout << "Frame: " << frames << std::endl;
    
    // 绘制参考球
    glColor3d(0, 0, 1);
    glutSolidSphere(0.97, 100, 100);
    
    // 调用 CUDA 模拟
    simulate_gpu(N, EPS, ITERS, BIAS);
    
    // 将设备端计算结果复制到 host 内存
    cudaMemcpy(reinterpret_cast<float3*>(vertices.data()), d_vertices, sizeof(float3) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(reinterpret_cast<float3*>(normals.data()), d_normals, sizeof(float3) * N * N, cudaMemcpyDeviceToHost);
    
    // 更新 OpenGL 缓冲区数据（顶点与法线）
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STREAM_DRAW);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glColorPointer(4, GL_FLOAT, 0, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * normals.size(), normals.data(), GL_STREAM_DRAW);
    glNormalPointer(GL_FLOAT, 0, 0);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLE_STRIP, indices.size(), GL_UNSIGNED_INT, 0);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);
    
    // 周期性重新初始化数据（比如重置模拟）
    if ((int)(frames * G) % 1000 == 0)
        init_data(N);
    
    glutSwapBuffers();
    glutPostRedisplay();
}

/**************************************************
 * 数据与 OpenGL 缓冲区初始化
 **************************************************/
void init_data(int n) {
    // 初始化 host 侧顶点数据：均匀分布在 x,y∈[-2,2]，z = 2 的平面上
    vertices.clear();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float x = i * 4.0f / (n - 1) - 2;
            float y = j * 4.0f / (n - 1) - 2;
            float z = 2.0f;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    
    // 初始化颜色，每个顶点设为红色 (RGBA)
    colors.clear();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            colors.push_back(1.0f); // R
            colors.push_back(0.0f); // G
            colors.push_back(0.0f); // B
            colors.push_back(0.9f); // A
        }
    }
    
    // 生成三角带 indices，用于绘制整个网格
    indices.clear();
    for (int i = 0; i < n - 1; i++) {
        int base = i * n;
        indices.push_back(base);
        for (int j = 0; j < n; j++) {
            indices.push_back(base + j);
            indices.push_back(base + j + n);
        }
        indices.push_back(base + 2 * n - 1);
    }
    
    // 初始化法线（先全部置为 0，后续由 CUDA 更新）
    normals.resize(3 * n * n, 0.0f);
    
    // 初始化 host 侧速度，每个顶点速度置 0
    std::vector<float> temp_vel(3 * n * n, 0.0f);
    
    // 将 host 数据 sizes 调整为正确规模：
    // vertices：3*n*n floats, colors：4*n*n floats, normals：3*n*n floats.
    
    // 计算约束参数（假设网格均匀）
    cnstr_two = vertices[3*n] - vertices[0];
    cnstr_dia = std::sqrt(2 * cnstr_two * cnstr_two);
    
    // 创建 OpenGL 缓冲区对象
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &cbo);
    glGenBuffers(1, &nbo);
    glGenBuffers(1, &ibo);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(),
                 vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * normals.size(),
                 normals.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * colors.size(),
                 colors.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(),
                 indices.data(), GL_STATIC_DRAW);
    
    // 分配设备侧内存，并拷贝初始数据
    if(d_vertices) cudaFree(d_vertices);
    cudaMalloc(&d_vertices, sizeof(float3) * n * n);
    cudaMemcpy(d_vertices, reinterpret_cast<float3*>(vertices.data()), sizeof(float3)*n*n, cudaMemcpyHostToDevice);
    
    if(d_velocities) cudaFree(d_velocities);
    cudaMalloc(&d_velocities, sizeof(float3) * n * n);
    cudaMemset(d_velocities, 0, sizeof(float3) * n * n);
    
    if(d_normals) cudaFree(d_normals);
    cudaMalloc(&d_normals, sizeof(float3) * n * n);
    cudaMemset(d_normals, 0, sizeof(float3) * n * n);
}

/**************************************************
 * OpenGL 初始化
 **************************************************/
void init_GL(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Position-Based Dynamics - CUDA (Explicit Copy)");
    glutDisplayFunc(display);
    
    glewInit();
    
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    GLfloat mat_specular[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    GLfloat mat_shininess[] = { 50.0f };
    GLfloat light_position[] = { 1.0f, 1.0f, 1.0f, 0.0f };
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
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);
    glTranslatef(0.0f, 0.0f, -6.0f);
    glRotated(300, 1, 0, 0);
    glRotated(270, 0, 0, 1);
}

int main(int argc, char** argv) {
    init_GL(&argc, argv);
    init_data(N);
    glutMainLoop();
    return 0;
}
