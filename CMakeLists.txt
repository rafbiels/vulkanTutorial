cmake_minimum_required(VERSION 3.10)

project(VulkanTutorial)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(stb REQUIRED)
find_package(Vulkan REQUIRED)

add_library(MyVulkanApp MyVulkanApp.cxx)
target_include_directories(MyVulkanApp
                           PUBLIC
                           ${GLFW_INCLUDE_DIR}
                           ${Vulkan_INCLUDE_DIRS}
                           PRIVATE
                           ${STB_INCLUDE_DIR})
target_link_libraries(MyVulkanApp
                      PUBLIC
                      ${GLFW_LIBRARY}
                      ${Vulkan_LIBRARIES}
                      glm::glm)

add_executable(app main.cxx)
target_link_libraries(app
                      PUBLIC
                      MyVulkanApp)

foreach(SHADER_NAME vert frag)
  add_custom_command(OUTPUT ${SHADER_NAME}.spv
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.${SHADER_NAME}
                     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
                     COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.${SHADER_NAME}
                             -o ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv)
endforeach()
add_custom_target(shaders ALL DEPENDS vert.spv frag.spv)
