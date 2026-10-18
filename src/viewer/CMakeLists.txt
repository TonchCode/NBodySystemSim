find_package(OpenGL REQUIRED)

# Dear ImGui 1.91.5 (third_party/imgui, D-032): core + GLFW and OpenGL3 backends, compiled into the viewer only.
# The OpenGL3 backend uses its own embedded loader (imgui_impl_opengl3_loader.h, "imgl3w"): its symbols are
# imgl3w*/imgl3wProcs, glad's are glad_gl*, and the backend TU never includes glad, so the two loaders coexist
# (each resolves its own function pointers from the same current context).
set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/third_party/imgui)
add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw)

add_executable(nbody_viewer main.cpp gl_loader.c viewer_kernels.cu)
target_include_directories(nbody_viewer PRIVATE ${PROJECT_SOURCE_DIR}/third_party/glad/include)
target_compile_options(nbody_viewer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(nbody_viewer PRIVATE nbody_core imgui glfw OpenGL::GL)
