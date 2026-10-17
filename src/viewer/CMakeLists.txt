find_package(OpenGL REQUIRED)

add_executable(nbody_viewer main.cpp gl_loader.c)
target_include_directories(nbody_viewer PRIVATE ${PROJECT_SOURCE_DIR}/third_party/glad/include)
target_link_libraries(nbody_viewer PRIVATE nbody_core glfw OpenGL::GL)
