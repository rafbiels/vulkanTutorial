cmake_minimum_required(VERSION 3.10)

project(VulkanTutorial)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

add_library(HelloTriangleApplication HelloTriangleApplication.cxx)
target_include_directories(HelloTriangleApplication
                           PUBLIC
                           ${GLFW_INCLUDE_DIR}
                           ${Vulkan_INCLUDE_DIRS})
target_link_libraries(HelloTriangleApplication
                      PUBLIC
                      ${GLFW_LIBRARY}
                      ${Vulkan_LIBRARIES}
                      glm::glm)

add_executable(myVulkanApp main.cxx)
target_link_libraries(myVulkanApp
                      PUBLIC
                      HelloTriangleApplication)
