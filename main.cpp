#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
  constexpr int c_windowWidth{800}, c_windowHeight{600};
  constexpr void throwOnFailure(vk::Result result, const std::string& msg="Failure!") {
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error(msg);
    }
  }
  constexpr void throwOnFailure(bool result, const std::string& msg="Failure!") {
    if (!result) {
      throw std::runtime_error(msg);
    }
  }
} // anonymous namespace

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    m_window = glfwCreateWindow(c_windowWidth, c_windowHeight, "Vulkan",
                                nullptr, nullptr);
  }

  void initVulkan() {
    createInstance();
  }

  void mainLoop() {
    while (glfwWindowShouldClose(m_window) == 0) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    vkDestroyInstance(m_instance, nullptr);
    glfwDestroyWindow(m_window);
    glfwTerminate();
  }

  void createInstance() {
    // -------------------------------------------
    // Initialise Vulkan
    // -------------------------------------------
    vk::ApplicationInfo appInfo = {
      .sType = vk::StructureType::eApplicationInfo,
      .pApplicationName = "Hello Triangle",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_2
    };

    uint32_t glfwExtensionCount{0};
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::cout << "required extensions:\n";
    const auto glfwExtensionsSpan = std::span{glfwExtensions,glfwExtensionCount};
    for (const auto& ext : glfwExtensionsSpan) {
      std::cout << '\t' << ext << '\n';
    }

    vk::InstanceCreateInfo createInfo = {
      .sType = vk::StructureType::eInstanceCreateInfo,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = 0,
      .enabledExtensionCount = glfwExtensionCount,
      .ppEnabledExtensionNames = glfwExtensions
    };
    if (vk::createInstance(&createInfo, nullptr, &m_instance) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create instance!");
    }

    // -------------------------------------------
    // Check for extensions
    // -------------------------------------------
    uint32_t extensionCount = 0;
    throwOnFailure(vk::enumerateInstanceExtensionProperties(
      nullptr, &extensionCount, nullptr));
    std::vector<vk::ExtensionProperties> extensions(extensionCount);
    throwOnFailure(vk::enumerateInstanceExtensionProperties(
      nullptr, &extensionCount, extensions.data()));
    std::cout << "available extensions:\n";
    for (const auto& extension : extensions) {
      std::cout << '\t' << extension.extensionName << '\n';
    }
    for (const auto& extName : glfwExtensionsSpan) {
      auto matchName = [&extName](const vk::ExtensionProperties& e){
        return std::string_view(e.extensionName) == std::string_view(extName);
      };
      throwOnFailure(std::ranges::find_if(extensions, matchName) != extensions.end(),
                     std::string("Required extension ") + extName + " is not available");
    }
  }

  GLFWwindow* m_window{nullptr};
  vk::Instance m_instance;
};

int main() {
  HelloTriangleApplication app{};

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
