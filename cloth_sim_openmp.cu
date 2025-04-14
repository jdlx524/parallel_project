// cloth_sim_cpu.cpp
#include <GL/glew.h>
#include <GL/glut.h>

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <omp.h>

// 自定义 float3 结构体，表示 3 个 float 分量
struct float3 {
    float x, y, z;
};

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

// 根据 HIQ 宏选择测试规模和迭代次数
#define HIQ
#ifdef HIQ
    #define N (128)
    #define EPS (1E-2)
    #define BIAS (0.15)
    #define ITERS (400)
    #define G (1)
#else
    #define N (64)
    #define EPS (1E-2)
    #define BIAS (0.17)
    #define ITERS (128)
    #define G (1)
#endif

/**************************************************
 * constants
 **************************************************/
const unsigned int window_width = 1920;
const unsigned int window_height = 1080;

// OpenGL VBO、IBO、CBO、NBO
static GLuint ibo = 0;
static GLuint vbo = 0;
static GLuint cbo = 0;
static GLuint nbo = 0;

// host-sided vectors，所有数据均以连续 float 数组存储，格式为 {x0,y0,z0, x1,y1,z1, ...}
static std::vector<float> vertices;
static std::vector<float> velocities;
static std::vector<float> colors;
static std::vector<float> normals;
static std::vector<int> indices;

// 约束参数，全局变量
static float cnstr_two;
static float cnstr_dia;

// 帧计数
static size_t frames = 0;

/**************************************************
 * 函数前置声明
 **************************************************/
void display();
void init_GL();
void init_data(int);

/**************************************************
 * CPU 物理计算辅助函数
 **************************************************/

// 更新单个顶点位置：如果点的距离平方大于 1，则在 z 方向上减小速度（类似一种“收缩”效果）
// 然后用速度更新位置（简单积分）
inline void update_positions(float3& pos, float3& vel, const float eps) {
    // 如果点距离原点平方大于 1，将其在 z 方向上施加一个微调
    if (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z > 1)
        vel.z -= eps * G;
    
    pos.x += eps * vel.x;
    pos.y += eps * vel.y;
    pos.z += eps * vel.z;
}

// 将顶点“贴”到球面上，若该点的距离小于 1，则保持不变；否则按比例缩放到距离为 1
inline void adjust_positions(float3& pos) {
    float len2 = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    float invrho = 1.0f / std::sqrt(len2);
    // 如果 invrho 小于 1 说明点距离 > 1，则缩放；否则保持不变
    pos.x *= (invrho < 1 ? invrho : 1);
    pos.y *= (invrho < 1 ? invrho : 1);
    pos.z *= (invrho < 1 ? invrho : 1);
}

// 对两个约束点进行修正，使它们的距离趋于目标 constraint
inline void relax_constraint(const float3* Pos, float3* Tmp,
                             const int l, const int m,
                             const float constraint, const float bias) {
    float3 delta;
    delta.x = Pos[l].x - Pos[m].x;
    delta.y = Pos[l].y - Pos[m].y;
    delta.z = Pos[l].z - Pos[m].z;
    
    float len2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
    float invlen = 1.0f / std::sqrt(len2);
    float factor = (1.0f - constraint * invlen) * bias;
    
    Tmp[l].x -= delta.x * factor;
    Tmp[l].y -= delta.y * factor;
    Tmp[l].z -= delta.z * factor;
    
    Tmp[m].x += delta.x * factor;
    Tmp[m].y += delta.y * factor;
    Tmp[m].z += delta.z * factor;
}

// 对 normal 进行归一化
inline void normalize(float3& normal) {
    float len2 = normal.x*normal.x + normal.y*normal.y + normal.z*normal.z;
    float invrho = 1.0f / std::sqrt(len2);
    normal.x *= invrho;
    normal.y *= invrho;
    normal.z *= invrho;
}

// 根据某个中心顶点和邻域顶点，计算法线的一部分，并累加到 Normal
inline void wedge(const float3* Vertices, float3& Normal,
                  const int& i, const int& j, const int n,
                  const int& a, const int& b) {
    float3 center = Vertices[i*n+j];
    float3 span_u = Vertices[(i+a)*n+j];
    float3 span_v = Vertices[i*n+(j+b)];
    
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

/**************************************************
 * CPU 版本核心计算（OpenMP 并行）
 **************************************************/

// propagate：对每个顶点更新位置（积分速度）
// vertices 与 velocities 都被视为 n*n 个 float3（排列为 3*n*n 个 float）
void propagate(float3* verts, float3* vels, const int n, const float eps = EPS) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            update_positions(verts[i*n+j], vels[i*n+j], eps);
        }
    }
}

// validate：对所有约束进行迭代修正
void validate(float3* verts, const int n, const int iters = ITERS, const float bias = BIAS) {
    // 使用双缓冲：temp 存储修正后的顶点位置
    std::vector<float3> temp(n*n);
    
    for (int iter = 0; iter < iters; iter++) {
        // 拷贝当前状态到 temp
        std::memcpy(temp.data(), verts, sizeof(float3)*n*n);
        
        // 对水平约束
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n; j++) {
                relax_constraint(verts, temp.data(), i*n + j, (i+1)*n + j, cnstr_two, bias);
            }
        }
        // 对垂直约束
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - 1; j++) {
                relax_constraint(verts, temp.data(), i*n + j, i*n + (j+1), cnstr_two, bias);
            }
        }
        // 对两格水平约束
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n - 2; i++) {
            for (int j = 0; j < n; j++) {
                relax_constraint(verts, temp.data(), i*n + j, (i+2)*n + j, 2*cnstr_two, bias);
            }
        }
        // 对两格垂直约束
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - 2; j++) {
                relax_constraint(verts, temp.data(), i*n + j, i*n + (j+2), 2*cnstr_two, bias);
            }
        }
        // 对斜对角约束1
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - 1; j++) {
                relax_constraint(verts, temp.data(), i*n + j, (i+1)*n + (j+1), cnstr_dia, bias);
            }
        }
        // 对斜对角约束2
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n - 1; j++) {
                relax_constraint(verts, temp.data(), i*n + j, (i-1)*n + (j+1), cnstr_dia, bias);
            }
        }
        // 对所有点做贴球面调整
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                adjust_positions(temp[i*n+j]);
            }
        }
        
        // 更新 verts
        std::memcpy(verts, temp.data(), sizeof(float3)*n*n);
    }
}

// 更新每个顶点的法线（利用周围 4 个点产生交叉乘）
void update_normals(float3* verts, float3* norms, const int n) {
    // 将 normals 数组置 0
    std::memset(norms, 0, sizeof(float3) * n * n);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float3 normal = {0, 0, 0};
            if (i > 0 && j > 0)
                wedge(verts, normal, i, j, n, -1, -1);
            if (i > 0 && j+1 < n)
                wedge(verts, normal, i, j, n, -1, +1);
            if (i+1 < n && j > 0)
                wedge(verts, normal, i, j, n, +1, -1);
            if (i+1 < n && j+1 < n)
                wedge(verts, normal, i, j, n, +1, +1);
            
            normalize(normal);
            norms[i*n + j] = normal;
        }
    }
}

/**************************************************
 * OpenGL 渲染与显示函数
 **************************************************/
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotated((frames++) * 0.2, 0, 0, 1);

    if (frames % 500 == 0) {
        std::cout << frames * 1000.0 / glutGet(GLUT_ELAPSED_TIME) << std::endl;
    }

    glColor3d(0, 0, 1);
    glutSolidSphere(0.97, 100, 100);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);

    // 更新物理仿真：采用 CPU 版本的 propagate 和 validate
    propagate(reinterpret_cast<float3*>(vertices.data()),
              reinterpret_cast<float3*>(velocities.data()),
              N, EPS);
    validate(reinterpret_cast<float3*>(vertices.data()), N);
    
    // 上传更新后的顶点数据和法线数据到 GPU
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(),
                 vertices.data(), GL_STREAM_DRAW);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    update_normals(reinterpret_cast<float3*>(vertices.data()),
                   reinterpret_cast<float3*>(normals.data()), N);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*normals.size(),
                 normals.data(), GL_STREAM_DRAW);
    glNormalPointer(GL_FLOAT, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLE_STRIP, indices.size(), GL_UNSIGNED_INT, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    // 周期性重新初始化数据（例如模拟重置）
    if ((int)(frames * G) % 1000 == 0)
        init_data(N);

    glutSwapBuffers();
    glutPostRedisplay();
}

/**************************************************
 * 数据与 OpenGL 缓冲区初始化
 **************************************************/
void init_data(int n) {
    // 初始化顶点：布置在 x,y ∈ [-2,2]，z = 2 的平面上
    vertices.clear();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float x = i * 4.0f / (n - 1) - 2;
            float y = j * 4.0f / (n - 1) - 2;
            float z = 2;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    // 初始化 velocities，每个顶点的速度置 0
    velocities.resize(3 * n * n, 0.0f);

    // 计算约束距离，cnstr_two 为相邻点距离（x 轴方向差值），cnstr_dia 为对角距离
    cnstr_two = vertices[3*n] - vertices[0];
    cnstr_dia = std::sqrt(2 * cnstr_two * cnstr_two);

    // 计算初始法线
    normals.resize(3 * n * n, 0.0f);
    update_normals(reinterpret_cast<float3*>(vertices.data()),
                   reinterpret_cast<float3*>(normals.data()), n);

    // 初始化颜色，每个顶点设为红色（RGBA）
    colors.clear();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            colors.push_back(1.0f);  // r
            colors.push_back(0.0f);  // g
            colors.push_back(0.0f);  // b
            colors.push_back(0.9f);  // a
        }
    }

    // 生成三角带 indices
    indices.clear();
    for (int i = 0; i < n - 1; i++) {
        int base = i * n;
        indices.push_back(base);
        for (int j = 0; j < n; j++) {
            indices.push_back(base + j);
            indices.push_back(base + j + n);
        }
        indices.push_back(base + 2*n - 1);
    }

    // 生成 OpenGL 缓冲区对象
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &cbo);
    glGenBuffers(1, &nbo);
    glGenBuffers(1, &ibo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(),
                 vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*normals.size(),
                 normals.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*colors.size(),
                 colors.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*indices.size(),
                 indices.data(), GL_STATIC_DRAW);
}

/**************************************************
 * OpenGL 初始化
 **************************************************/
void init_GL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Position-Based Dynamics - CPU (OpenMP)");
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

int main(int argc, char **argv) {
    init_GL(&argc, argv);
    init_data(N);
    glutMainLoop();
    return 0;
}
