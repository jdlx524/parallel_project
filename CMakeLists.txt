cmake_minimum_required(VERSION 3.22)
project(parallel_project LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 OpenGL 和 GLUT
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# 添加可执行文件
add_executable(parallel_project cloth_sim.cu)

# 关键：确保 NVCC 正确编译 OpenGL/GLUT 代码
set_target_properties(parallel_project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75 # 根据 GPU 选择，75 对应 Turing
)

# 链接 OpenGL 和 GLUT 库
target_link_libraries(parallel_project PRIVATE
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        cudart
)

# 让 NVCC 识别 C++ 代码（特别是 OpenGL/GLUT）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
