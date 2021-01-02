#ifndef HELLOTRIANGLEAPPLICATION_H
#define HELLOTRIANGLEAPPLICATION_H

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <optional>
#include <vector>

class HelloTriangleApplication {
public:
  // --------------- public methods ------------------------
  void run();
private:
  // --------------- private types -------------------------
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    [[nodiscard]] constexpr bool isComplete() const {
        return graphicsFamily.has_value() &&
               presentFamily.has_value();
    }
  };
  struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
  };
  // --------------- private methods -----------------------
  void initWindow();
  void initVulkan();
  void mainLoop();
  void cleanup();
  void createInstance();
  void setupDebugMessenger();
  void cleanupDebugMessenger();
  void createSurface();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapChain();
  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
  bool isDeviceSuitable(vk::PhysicalDevice device);
  SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

  // --------------- static private methods ----------------
  static bool checkValidationLayerSupport();
  static bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
  static std::vector<const char*> getRequiredExtensions();
  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats);
  static vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes);
  static vk::DebugUtilsMessengerCreateInfoEXT createDebugMessengerCreateInfo();

  // Cannot use c++ types as arguments because API requires this function signature
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData);

  // --------------- private members -----------------------
  GLFWwindow* m_window{nullptr};
  vk::Instance m_instance;
  vk::DispatchLoaderDynamic m_dispatchLoaderDynamic;
  vk::DebugUtilsMessengerEXT m_debugMessenger;
  vk::SurfaceKHR m_surface;
  vk::PhysicalDevice m_physicalDevice;
  vk::Device m_device;
  vk::Queue m_graphicsQueue;
  vk::Queue m_presentQueue;
  vk::SwapchainKHR m_swapChain;
  std::vector<vk::Image> m_swapChainImages;
  vk::Format m_swapChainImageFormat;
  vk::Extent2D m_swapChainExtent;
}; // class HelloTriangleApplication

#endif // HELLOTRIANGLEAPPLICATION_H
