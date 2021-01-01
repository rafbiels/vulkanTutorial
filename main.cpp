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
  // ---------------------------------------------------------------------------

  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    m_window = glfwCreateWindow(c_windowWidth, c_windowHeight, "Vulkan",
                                nullptr, nullptr);
  }

  // ---------------------------------------------------------------------------

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
  }

  // ---------------------------------------------------------------------------

  void mainLoop() {
    while (glfwWindowShouldClose(m_window) == 0) {
      glfwPollEvents();
    }
  }

  // ---------------------------------------------------------------------------

  void cleanup() {
    cleanupDebugMessenger();
    m_instance.destroy();
    glfwDestroyWindow(m_window);
    glfwTerminate();
  }

  // ---------------------------------------------------------------------------

  void createInstance() {
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

    const std::vector<const char*> extensions = getRequiredExtensions();

    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (c_enableValidationLayers) {
      debugCreateInfo = createDebugMessengerCreateInfo();
    };
    vk::InstanceCreateInfo createInfo = {
      .sType = vk::StructureType::eInstanceCreateInfo,
      .pNext = (c_enableValidationLayers ? &debugCreateInfo : nullptr),
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(c_enableValidationLayers ? c_validationLayers.size() : 0),
      .ppEnabledLayerNames = (c_enableValidationLayers ? c_validationLayers.data() : nullptr),
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()
    };
    if (vk::createInstance(&createInfo, nullptr, &m_instance) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create instance!");
    }

    m_dispatchLoaderDynamic = vk::DispatchLoaderDynamic{m_instance, vkGetInstanceProcAddr};
  }

  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------

  static std::vector<const char*> getRequiredExtensions() {
    // -------------------------------------------
    // Build the list of required extensions
    // -------------------------------------------
    uint32_t glfwExtensionCount{0};
    const char** glfwExtensions{nullptr};
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    const auto glfwExtensionsSpan = std::span{glfwExtensions, glfwExtensionCount};

    std::vector<const char*> requiredExtensions(glfwExtensionsSpan.begin(), glfwExtensionsSpan.end());

    if (c_enableValidationLayers) {
      requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::cout << "required extensions:" << std::endl;
    for (const auto& ext : requiredExtensions) {
      std::cout << '\t' << ext << std::endl;
    }

    // -------------------------------------------
    // Check if the extensions are available
    // -------------------------------------------
    uint32_t availableExtensionCount = 0;
    throwOnFailure(vk::enumerateInstanceExtensionProperties(
      nullptr, &availableExtensionCount, nullptr));
    std::vector<vk::ExtensionProperties> availableExtensions(availableExtensionCount);
    throwOnFailure(vk::enumerateInstanceExtensionProperties(
      nullptr, &availableExtensionCount, availableExtensions.data()));

    std::cout << "available extensions:" << std::endl;
    for (const auto& ext : availableExtensions) {
      std::cout << '\t' << ext.extensionName << std::endl;
    }

    for (const auto& extName : requiredExtensions) {
      auto matchName = [&extName](const vk::ExtensionProperties& e){
        return std::string_view(e.extensionName) == std::string_view(extName);
      };
      throwOnFailure(
        std::ranges::find_if(availableExtensions, matchName) != availableExtensions.end(),
        std::string("Required extension ") + extName + " is not available");
    }

    return requiredExtensions;
  }

  // ---------------------------------------------------------------------------

  // Cannot use c++ types as arguments because API requires this function signature
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
    VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
  }

  // ---------------------------------------------------------------------------

  static vk::DebugUtilsMessengerCreateInfoEXT createDebugMessengerCreateInfo() {
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    return {
      .sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
      .messageSeverity = Severity::eVerbose | Severity::eInfo |
                         Severity::eWarning | Severity::eError,
      .messageType = Type::eGeneral | Type::eValidation | Type::ePerformance,
      .pfnUserCallback = debugCallback,
      .pUserData = nullptr
    };
  }

  // ---------------------------------------------------------------------------

  void setupDebugMessenger() {
    if (!c_enableValidationLayers) {return;}
    vk::DebugUtilsMessengerCreateInfoEXT createInfo = createDebugMessengerCreateInfo();
    throwOnFailure(m_instance.createDebugUtilsMessengerEXT(
      &createInfo, nullptr, &m_debugMessenger, m_dispatchLoaderDynamic));
  }

  // ---------------------------------------------------------------------------

  void cleanupDebugMessenger() {
    if (!c_enableValidationLayers) {return;}
    m_instance.destroyDebugUtilsMessengerEXT(m_debugMessenger, nullptr, m_dispatchLoaderDynamic);
  }

  // ---------------------------------------------------------------------------

  GLFWwindow* m_window{nullptr};
  vk::Instance m_instance;
  vk::DispatchLoaderDynamic m_dispatchLoaderDynamic;
  vk::DebugUtilsMessengerEXT m_debugMessenger;
}; // class HelloTriangleApplication

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
