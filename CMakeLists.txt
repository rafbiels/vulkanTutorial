cmake_minimum_required(VERSION 3.10)

project(TestVulkan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(myVulkanApp main.cpp)
target_include_directories(myVulkanApp
                           PUBLIC
                           ${GLFW_INCLUDE_DIR}
                           ${Vulkan_INCLUDE_DIRS})
target_link_libraries(myVulkanApp
                      PUBLIC
                      ${GLFW_LIBRARY}
                      ${Vulkan_LIBRARIES}
                      glm::glm)
