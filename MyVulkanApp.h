#ifndef MyVulkanApp_H
#define MyVulkanApp_H

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <chrono>
#include <optional>
#include <utility>
#include <vector>

class MyVulkanApp {
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
  struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
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
  void createImageViews();
  void createRenderPass();
  void createGraphicsPipeline();
  void createFramebuffers();
  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createCommandPool();
  void createCommandBuffers();
  void createTextureImage();
  void createTextureImageView();
  void createTextureSampler();
  void createSyncObjects();
  void cleanupSwapChain();
  void recreateSwapChain();
  void drawFrame();
  vk::ShaderModule createShaderModule(std::vector<char> code);
  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
  bool isDeviceSuitable(vk::PhysicalDevice device);
  SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
  void setFramebufferResized(bool value) {m_framebufferResized = value;}
  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
  std::pair<vk::Buffer, vk::DeviceMemory> createBuffer(
    vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);
  std::pair<vk::CommandPool, vk::CommandBuffer> beginSingleTimeCommands();
  void endSingleTimeCommands(vk::CommandPool commandPool, vk::CommandBuffer commandBuffer);
  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
  void updateUniformBuffer(uint32_t currentImage);
  std::pair<vk::Image, vk::DeviceMemory> createImage(
    uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
    vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties);
  void transitionImageLayout(
    vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
  void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
  vk::ImageView createImageView(vk::Image image, vk::Format format);

  // --------------- static private methods ----------------
  static bool checkValidationLayerSupport();
  static bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
  static std::vector<const char*> getRequiredExtensions();
  static std::vector<char> readFile(const std::string& filename);
  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats);
  static vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes);
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
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
  std::vector<vk::ImageView> m_swapChainImageViews;
  vk::RenderPass m_renderPass;
  vk::PipelineLayout m_pipelineLayout;
  vk::Pipeline m_graphicsPipeline;
  std::vector<vk::Framebuffer> m_swapChainFramebuffers;
  vk::CommandPool m_commandPool;
  std::vector<vk::CommandBuffer> m_commandBuffers;
  std::vector<vk::Semaphore> m_imageAvailableSemaphores;
  std::vector<vk::Semaphore> m_renderFinishedSemaphores;
  std::vector<vk::Fence> m_inFlightFences;
  std::vector<vk::Fence> m_imagesInFlight;
  vk::Buffer m_vertexBuffer;
  vk::DeviceMemory m_vertexBufferMemory;
  vk::Buffer m_indexBuffer;
  vk::DeviceMemory m_indexBufferMemory;
  std::vector<vk::Buffer> m_uniformBuffers;
  std::vector<vk::DeviceMemory> m_uniformBuffersMemory;
  vk::DescriptorPool m_descriptorPool;
  vk::DescriptorSetLayout m_descriptorSetLayout;
  std::vector<vk::DescriptorSet> m_descriptorSets;
  vk::Image m_textureImage;
  vk::DeviceMemory m_textureImageMemory;
  vk::ImageView m_textureImageView;
  vk::Sampler m_textureSampler;
  bool m_framebufferResized{false};
}; // class MyVulkanApp

class FpsCounter {
public:
  using Clock_t = std::chrono::steady_clock;
  explicit FpsCounter(size_t maxCount) : m_maxFrameNumber(maxCount) {}
  FpsCounter& operator++();
private:
  size_t m_frameNumber{0};
  size_t m_maxFrameNumber;
  std::chrono::time_point<Clock_t> m_lastTimestamp{};
};

struct Vertex {
  glm::vec2 pos;
  glm::vec3 colour;
  glm::vec2 texCoord;
  static vk::VertexInputBindingDescription getBindingDescription();
  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions();
};

#endif // MyVulkanApp_H
