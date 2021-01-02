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
    [[nodiscard]] constexpr bool isComplete() const {
        return graphicsFamily.has_value();
    }
  };
  // --------------- private methods -----------------------
  void initWindow();
  void initVulkan();
  void mainLoop();
  void cleanup();
  void createInstance();
  void setupDebugMessenger();
  void cleanupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();

  // --------------- static private methods ----------------
  static QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
  static bool isDeviceSuitable(vk::PhysicalDevice device);
  static bool checkValidationLayerSupport();
  static std::vector<const char*> getRequiredExtensions();
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
  vk::PhysicalDevice m_physicalDevice;
  vk::Device m_device;
  vk::Queue m_graphicsQueue;
}; // class HelloTriangleApplication

#endif // HELLOTRIANGLEAPPLICATION_H
