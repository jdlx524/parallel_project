#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <cuda_runtime.h>
#include <GL/glut.h> // OpenGL for 3D visualization

#define WIDTH 800
#define HEIGHT 600
#define GRID_SIZE 20 // Grid size (NxN points)
#define POINT_SPACING 20 // Distance between points
#define GRAVITY 0.1 // Gravity force
#define DAMPING 0.99 // Energy loss factor
#define POINT_RADIUS 3 // Size of points for visualization

// Structure to represent a mass point in 3D
struct Point {
    float x, y, z;  // Current position in 3D
    float prev_x, prev_y, prev_z; // Previous position for velocity calculation
    bool fixed; // Whether this point is fixed in space
    float force_x, force_y, force_z; // Accumulated force

    // Default constructor
    Point() : x(0), y(0), z(0), prev_x(0), prev_y(0), prev_z(0), fixed(false), force_x(0), force_y(0), force_z(0) {}

    // Parameterized constructor
    Point(float x, float y, float z, bool fixed=false)
            : x(x), y(y), z(z), prev_x(x), prev_y(y), prev_z(z), fixed(fixed),
              force_x(0), force_y(0), force_z(0) {}
};

// Grid of mass points
std::vector<std::vector<Point>> points(GRID_SIZE, std::vector<Point>(GRID_SIZE));

// Initialize the grid of points
void initializeGrid() {
#pragma omp parallel for collapse(2) // Parallelize initialization
    for (int j = 0; j < GRID_SIZE; j++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            bool fixed = (j == 0 && i % 3 == 0); // Fix some top-row points
            points[j][i] = Point(100 + i * POINT_SPACING, 100 + j * POINT_SPACING, 0, fixed);
            points[j][i].z = sin(i * 0.2) * 20; // 让 Z 轴有变化
        }
    }
}

// CUDA Kernel to update forces
__global__ void applyGravity(Point* d_points, int gridSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= gridSize || idy >= gridSize) return;

    int index = idy * gridSize + idx;
    if (!d_points[index].fixed) {
        d_points[index].force_y += GRAVITY;
    }
}

// Function to apply forces (OpenMP for CPU, CUDA for GPU)
void applyForces() {
#ifdef USE_CUDA
    Point* d_points;
    cudaMalloc(&d_points, GRID_SIZE * GRID_SIZE * sizeof(Point));
    cudaMemcpy(d_points, points.data(), GRID_SIZE * GRID_SIZE * sizeof(Point), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((GRID_SIZE + 15) / 16, (GRID_SIZE + 15) / 16);
    applyGravity<<<numBlocks, threadsPerBlock>>>(d_points, GRID_SIZE);
    cudaMemcpy(points.data(), d_points, GRID_SIZE * GRID_SIZE * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaFree(d_points);
#else
#pragma omp parallel for collapse(2)
    for (int j = 0; j < GRID_SIZE; j++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            if (!points[j][i].fixed) {
                points[j][i].force_y += GRAVITY;
            }
        }
    }
#endif
}

// Function to update positions based on velocity
void updatePositions() {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < GRID_SIZE; j++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            if (!points[j][i].fixed) {
                float velocity_x = (points[j][i].x - points[j][i].prev_x) * DAMPING;
                float velocity_y = (points[j][i].y - points[j][i].prev_y) * DAMPING;
                float velocity_z = (points[j][i].z - points[j][i].prev_z) * DAMPING;
                points[j][i].prev_x = points[j][i].x;
                points[j][i].prev_y = points[j][i].y;
                points[j][i].prev_z = points[j][i].z;
                points[j][i].x += velocity_x + points[j][i].force_x;
                points[j][i].y += velocity_y + points[j][i].force_y;
                points[j][i].z += velocity_z + points[j][i].force_z;
                points[j][i].force_x = 0;
                points[j][i].force_y = 0;
                points[j][i].force_z = 0;
            }
        }
    }
}

// OpenGL function to render the cloth in 3D
void render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 设置背景为白色
    glLoadIdentity();
    gluLookAt(300, 300, -800, 300, 300, 300, 0, 1, 0);

    glEnable(GL_DEPTH_TEST);  // 确保深度测试打开
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPointSize(10.0f); // 让点更大
    glColor3f(1.0, 0.0, 0.0); // 红点
    glBegin(GL_POINTS);
    for (int j = 0; j < GRID_SIZE; j++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            glVertex3f(points[j][i].x, points[j][i].y, points[j][i].z);
        }
    }
    glEnd();

    std::cout << "Point(" << points[GRID_SIZE/2][GRID_SIZE/2].x
              << ", " << points[GRID_SIZE/2][GRID_SIZE/2].y
              << ", " << points[GRID_SIZE/2][GRID_SIZE/2].z << ")\n";

    glutSwapBuffers();
    glFlush();
}

// Simulation step
void simulate() {
    applyForces();
    updatePositions();
    glutPostRedisplay();
}

// OpenGL display loop
void display() {
    simulate();
    render();
}

// Main function
int main(int argc, char** argv) {
    initializeGrid();
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("3D Cloth Simulation");

    // 设置视口
    glViewport(0, 0, WIDTH, HEIGHT);

    // ✅ 设置透视投影矩阵
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)WIDTH / HEIGHT, 1.0, 2000.0);  // fov, aspect, near, far

    // ✅ 设置模型视图矩阵（观察矩阵）
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // ✅ 启用基本功能
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutMainLoop();
    return 0;
}
