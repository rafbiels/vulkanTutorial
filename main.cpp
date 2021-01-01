#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
  // Constants
  constexpr int c_windowWidth{800}, c_windowHeight{600};
  constexpr std::array<const char*,1> c_validationLayers = {
    "VK_LAYER_KHRONOS_validation"
  };
  #ifdef NDEBUG
    constexpr bool c_enableValidationLayers = false;
  #else
    constexpr bool c_enableValidationLayers = true;
  #endif
  // Helper functions
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
    if (c_enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error("Validation layers requested, but not available");
    }

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
      .enabledLayerCount = static_cast<uint32_t>(c_enableValidationLayers ? c_validationLayers.size() : 0),
      .ppEnabledLayerNames = (c_enableValidationLayers ? c_validationLayers.data() : nullptr),
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

  static bool checkValidationLayerSupport() {
    size_t notFound{0};
    uint32_t layerCount{0};
    throwOnFailure(vk::enumerateInstanceLayerProperties(
      &layerCount, nullptr));
    std::vector<vk::LayerProperties> availableLayers(layerCount);
    throwOnFailure(vk::enumerateInstanceLayerProperties(
      &layerCount, availableLayers.data()));
    std::cout << "required validation layers:" << std::endl;
    for (const auto& layerName : c_validationLayers) {
      std::cout << '\t' << layerName << std::endl;
    }
    std::cout << "available validation layers:" << std::endl;
    for (const auto& layer : availableLayers) {
      std::cout << '\t' << layer.layerName << std::endl;
    }
    for (const auto& layerName : std::span{c_validationLayers}) {
      auto matchName = [&layerName](const vk::LayerProperties& vl){
        return std::string_view(vl.layerName) == std::string_view(layerName);
      };
      if (std::ranges::find_if(availableLayers, matchName) == availableLayers.end()) {
        std::cout << "Validation layer " << layerName << " is not available" << std::endl;
        ++notFound;
      }
    }

    return notFound == 0;
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
