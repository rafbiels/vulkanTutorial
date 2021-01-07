#include "HelloTriangleApplication.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
  // Constants
  constexpr int c_windowWidth{800}, c_windowHeight{600};
  constexpr size_t c_maxFramesInFlight{2};
  constexpr size_t c_fpsCounterInterval{500};
  constexpr std::array<const char*,1> c_validationLayers = {
    "VK_LAYER_KHRONOS_validation"
  };
  constexpr std::array<const char*,1> c_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };
  #ifdef NDEBUG
    constexpr bool c_enableValidationLayers = false;
  #else
    constexpr bool c_enableValidationLayers = true;
  #endif
  constexpr std::array<Vertex,4> c_vertices = {
    Vertex{glm::vec2{-0.5F, -0.5F}, glm::vec3{1.0F, 0.0F, 0.0F}},
    Vertex{glm::vec2{ 0.5F, -0.5F}, glm::vec3{0.0F, 1.0F, 0.0F}},
    Vertex{glm::vec2{ 0.5F,  0.5F}, glm::vec3{0.0F, 0.0F, 1.0F}},
    Vertex{glm::vec2{-0.5F,  0.5F}, glm::vec3{1.0F, 1.0F, 1.0F}}
  };
  constexpr std::array<uint16_t, 6> c_indices = {0, 1, 2, 2, 3, 0};
  // Helper functions
  constexpr void throwOnFailure(bool result, const std::string& msg="Failure!") {
    if (!result) {
      throw std::runtime_error(msg);
    }
  }
  constexpr void throwOnFailure(vk::Result result, const std::string& msg="Failure!") {
    throwOnFailure(result==vk::Result::eSuccess, msg);
  }
} // anonymous namespace

// -----------------------------------------------------------------------------
void HelloTriangleApplication::run() {
  initWindow();
  initVulkan();
  mainLoop();
  cleanup();
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  m_window = glfwCreateWindow(c_windowWidth, c_windowHeight, "Vulkan",
                              nullptr, nullptr);
  glfwSetWindowUserPointer(m_window, this);
  glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::initVulkan() {
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandPool();
  createVertexBuffer();
  createIndexBuffer();
  createCommandBuffers();
  createSyncObjects();
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::mainLoop() {
  while (glfwWindowShouldClose(m_window) == 0) {
    glfwPollEvents();
    drawFrame();
  }
  m_device.waitIdle();
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::cleanup() {
  cleanupSwapChain();

  for (vk::Fence fence : m_inFlightFences) {
    m_device.destroyFence(fence);
  }
  m_inFlightFences.clear();
  for (vk::Semaphore semaphore : m_renderFinishedSemaphores) {
    m_device.destroySemaphore(semaphore);
  }
  m_renderFinishedSemaphores.clear();
  for (vk::Semaphore semaphore : m_imageAvailableSemaphores) {
    m_device.destroySemaphore(semaphore);
  }
  m_imageAvailableSemaphores.clear();
  m_device.destroyBuffer(m_vertexBuffer);
  m_device.freeMemory(m_vertexBufferMemory);
  m_device.destroyBuffer(m_indexBuffer);
  m_device.freeMemory(m_indexBufferMemory);
  m_device.destroyCommandPool(m_commandPool);
  m_device.destroy();
  cleanupDebugMessenger();
  m_instance.destroySurfaceKHR(m_surface);
  m_instance.destroy();
  glfwDestroyWindow(m_window);
  glfwTerminate();
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createInstance() {
  if (c_enableValidationLayers && !checkValidationLayerSupport()) {
    throw std::runtime_error("Validation layers requested, but not available");
  }

  vk::ApplicationInfo appInfo = {
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
    .pNext = (c_enableValidationLayers ? &debugCreateInfo : nullptr),
    .pApplicationInfo = &appInfo,
    .enabledLayerCount = static_cast<uint32_t>(c_enableValidationLayers ?
                                               c_validationLayers.size() : 0),
    .ppEnabledLayerNames = (c_enableValidationLayers ?
                            c_validationLayers.data() : nullptr),
    .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
    .ppEnabledExtensionNames = extensions.data()
  };
  throwOnFailure(vk::createInstance(&createInfo, nullptr, &m_instance),
                 "Failed to create instance");

  m_dispatchLoaderDynamic = vk::DispatchLoaderDynamic{m_instance, vkGetInstanceProcAddr};
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::setupDebugMessenger() {
  if (!c_enableValidationLayers) {return;}
  vk::DebugUtilsMessengerCreateInfoEXT createInfo = createDebugMessengerCreateInfo();
  throwOnFailure(m_instance.createDebugUtilsMessengerEXT(
    &createInfo, nullptr, &m_debugMessenger, m_dispatchLoaderDynamic));
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::cleanupDebugMessenger() {
  if (!c_enableValidationLayers) {return;}
  m_instance.destroyDebugUtilsMessengerEXT(m_debugMessenger, nullptr, m_dispatchLoaderDynamic);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createSurface() {
  vk::SurfaceKHR::CType surface{};
  throwOnFailure(static_cast<vk::Result>(
                   glfwCreateWindowSurface(m_instance, m_window, nullptr, &surface)),
                 "Failed to create window surface");
  m_surface = surface;
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  throwOnFailure(m_instance.enumeratePhysicalDevices(&deviceCount, nullptr));
  throwOnFailure(deviceCount>0, "Failed to find GPUs with Vulkan support");
  std::vector<vk::PhysicalDevice> devices(deviceCount);
  throwOnFailure(m_instance.enumeratePhysicalDevices(&deviceCount, devices.data()));
  std::cout << "available physical devices:" << std::endl;
  for (const auto& dev : devices) {
    std::cout << '\t' << dev.getProperties().deviceName << std::endl;
  }
  auto suitable = [this](vk::PhysicalDevice d){return isDeviceSuitable(d);};
  auto devIterator = std::ranges::find_if(devices, suitable);
  throwOnFailure(devIterator!=devices.end(), "Failed to find a suitable GPU");
  m_physicalDevice = *devIterator;
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
  std::set<uint32_t> uniqueQueueFamilies = {
    indices.graphicsFamily.value(),
    indices.presentFamily.value()
  };
  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
  float queuePriority = 1.0F;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo = {
        .queueFamilyIndex = queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
      };
      queueCreateInfos.push_back(queueCreateInfo);
  }

  vk::PhysicalDeviceFeatures deviceFeatures{};
  vk::DeviceCreateInfo createInfo = {
    .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
    .pQueueCreateInfos = queueCreateInfos.data(),
    .enabledExtensionCount = static_cast<uint32_t>(c_deviceExtensions.size()),
    .ppEnabledExtensionNames = c_deviceExtensions.data(),
    .pEnabledFeatures = &deviceFeatures
  };
  throwOnFailure(m_physicalDevice.createDevice(&createInfo, nullptr, &m_device),
                 "Failed to create logical device");
  m_device.getQueue(indices.graphicsFamily.value(), 0, &m_graphicsQueue);
  m_device.getQueue(indices.presentFamily.value(), 0, &m_presentQueue);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createSwapChain() {
  SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_physicalDevice);

  vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
  m_swapChainImageFormat = surfaceFormat.format;

  vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

  m_swapChainExtent = chooseSwapExtent(swapChainSupport.capabilities);
  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0) {
    imageCount = std::min(imageCount, swapChainSupport.capabilities.maxImageCount);
  }

  QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
  bool singleQueue = (indices.graphicsFamily == indices.presentFamily);
  auto queueFamilyIndices = std::array<uint32_t,2>{indices.graphicsFamily.value(),
                                                   indices.presentFamily.value()};

  vk::SwapchainCreateInfoKHR createInfo = {
    .surface = m_surface,
    .minImageCount = imageCount,
    .imageFormat = surfaceFormat.format,
    .imageColorSpace = surfaceFormat.colorSpace,
    .imageExtent = m_swapChainExtent,
    .imageArrayLayers = 1,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
    .imageSharingMode = (singleQueue ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent),
    .queueFamilyIndexCount = static_cast<uint32_t>(singleQueue ? 0 : 2),
    .pQueueFamilyIndices = (singleQueue ? nullptr : queueFamilyIndices.data()),
    .preTransform = swapChainSupport.capabilities.currentTransform,
    .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
    .presentMode = presentMode,
    .clipped = VK_TRUE,
    .oldSwapchain = nullptr
  };

  m_swapChain = m_device.createSwapchainKHR(createInfo);
  m_swapChainImages = m_device.getSwapchainImagesKHR(m_swapChain);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createImageViews() {
  m_swapChainImageViews.reserve(m_swapChainImages.size());
  for (const vk::Image image : m_swapChainImages) {
    vk::ImageViewCreateInfo createInfo = {
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = m_swapChainImageFormat,
      .components = {
        .r = vk::ComponentSwizzle::eIdentity,
        .g = vk::ComponentSwizzle::eIdentity,
        .b = vk::ComponentSwizzle::eIdentity,
        .a = vk::ComponentSwizzle::eIdentity
      },
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    m_swapChainImageViews.push_back(m_device.createImageView(createInfo));
  }
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createRenderPass() {
  vk::AttachmentDescription colourAttachment = {
    .format = m_swapChainImageFormat,
    .samples = vk::SampleCountFlagBits::e1,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eStore,
    .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
    .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
    .initialLayout = vk::ImageLayout::eUndefined,
    .finalLayout = vk::ImageLayout::ePresentSrcKHR
  };
  vk::AttachmentReference colourAttachmentRef = {
    .attachment = 0,
    .layout = vk::ImageLayout::eColorAttachmentOptimal
  };
  vk::SubpassDescription subpass = {
    .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
    .colorAttachmentCount = 1,
    .pColorAttachments = &colourAttachmentRef
  };
  vk::SubpassDependency dependency = {
    .srcSubpass = VK_SUBPASS_EXTERNAL,
    .dstSubpass = 0,
    .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
    .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
    .srcAccessMask = {},
    .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
  };
  vk::RenderPassCreateInfo renderPassInfo = {
    .attachmentCount = 1,
    .pAttachments = &colourAttachment,
    .subpassCount = 1,
    .pSubpasses = &subpass,
    .dependencyCount = 1,
    .pDependencies = &dependency
  };
  m_renderPass = m_device.createRenderPass(renderPassInfo);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createGraphicsPipeline() {
  std::vector<char> vertShaderCode = readFile("shaders/vert.spv");
  std::vector<char> fragShaderCode = readFile("shaders/frag.spv");

  // ---------------------------------------------
  // Shader modules
  // ---------------------------------------------
  vk::ShaderModule vertShaderModule = createShaderModule(std::move(vertShaderCode));
  vk::ShaderModule fragShaderModule = createShaderModule(std::move(fragShaderCode));

  vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {
    .stage = vk::ShaderStageFlagBits::eVertex,
    .module = vertShaderModule,
    .pName = "main"
  };
  vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {
    .stage = vk::ShaderStageFlagBits::eFragment,
    .module = fragShaderModule,
    .pName = "main"
  };

  auto shaderStages = std::array{vertShaderStageInfo, fragShaderStageInfo};

  // ---------------------------------------------
  // Fixed-function inputs
  // ---------------------------------------------
  // Vertex input assembly
  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();
  vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &bindingDescription,
    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
    .pVertexAttributeDescriptions = attributeDescriptions.data()
  };
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {
    .topology = vk::PrimitiveTopology::eTriangleList,
    .primitiveRestartEnable = VK_FALSE
  };

  // Viewports and scissors
  vk::Viewport viewport = {
    .x = 0.0F,
    .y = 0.0F,
    .width = static_cast<float>(m_swapChainExtent.width),
    .height = static_cast<float>(m_swapChainExtent.height),
    .minDepth = 0.0F,
    .maxDepth = 1.0F
  };
  vk::Rect2D scissor = {
    .offset = {0, 0},
    .extent = m_swapChainExtent
  };
  vk::PipelineViewportStateCreateInfo viewportState = {
    .viewportCount = 1,
    .pViewports = &viewport,
    .scissorCount = 1,
    .pScissors = &scissor
  };

  // Rasteriser
  vk::PipelineRasterizationStateCreateInfo rasteriser = {
    .depthClampEnable = VK_FALSE,
    .rasterizerDiscardEnable = VK_FALSE,
    .polygonMode = vk::PolygonMode::eFill,
    .cullMode = vk::CullModeFlagBits::eBack,
    .frontFace = vk::FrontFace::eClockwise,
    .depthBiasEnable = VK_FALSE,
    .lineWidth = 1.0F
  };
  vk::PipelineMultisampleStateCreateInfo multisampling = {
    .rasterizationSamples = vk::SampleCountFlagBits::e1,
    .sampleShadingEnable = VK_FALSE,
  };

  // Colour blending
  vk::PipelineColorBlendAttachmentState colourBlendAttachment = {
    .blendEnable = VK_FALSE,
    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
  };
  vk::PipelineColorBlendStateCreateInfo colourBlending = {
    .logicOpEnable = VK_FALSE,
    .attachmentCount = 1,
    .pAttachments = &colourBlendAttachment
  };

  // ---------------------------------------------
  // Pipeline layout
  // ---------------------------------------------
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {
    .setLayoutCount = 0,
    .pushConstantRangeCount = 0
  };
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo);

  // ---------------------------------------------
  // Create the pipeline
  // ---------------------------------------------
  vk::GraphicsPipelineCreateInfo pipelineInfo = {
    .stageCount = 2,
    .pStages = shaderStages.data(),
    .pVertexInputState = &vertexInputInfo,
    .pInputAssemblyState = &inputAssembly,
    .pViewportState = &viewportState,
    .pRasterizationState = &rasteriser,
    .pMultisampleState = &multisampling,
    .pColorBlendState = &colourBlending,
    .layout = m_pipelineLayout,
    .renderPass = m_renderPass,
    .subpass = 0
  };
  m_graphicsPipeline = m_device.createGraphicsPipeline(nullptr, pipelineInfo).value;

  // ---------------------------------------------
  // Clean-up
  // ---------------------------------------------
  m_device.destroyShaderModule(vertShaderModule);
  m_device.destroyShaderModule(fragShaderModule);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createFramebuffers() {
  for (const vk::ImageView& imageView : m_swapChainImageViews) {
    vk::FramebufferCreateInfo framebufferInfo = {
      .renderPass = m_renderPass,
      .attachmentCount = 1,
      .pAttachments = &imageView,
      .width = m_swapChainExtent.width,
      .height = m_swapChainExtent.height,
      .layers = 1
    };
    m_swapChainFramebuffers.push_back(m_device.createFramebuffer(framebufferInfo));
  }
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createVertexBuffer() {

  // Create staging buffer
  constexpr vk::DeviceSize bufferSize = sizeof(c_vertices[0]) * c_vertices.size();
  vk::MemoryPropertyFlags memFlags = vk::MemoryPropertyFlagBits::eHostVisible |
                                     vk::MemoryPropertyFlagBits::eHostCoherent;

  auto [stagingBuffer, stagingBufferMemory] =
    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent);

  // Copy data to staging buffer
  std::byte* pBufferMemory{nullptr};
  void** ppBufferMemory = reinterpret_cast<void**>(&pBufferMemory);
  throwOnFailure(
    m_device.mapMemory(stagingBufferMemory, 0, bufferSize, {}, ppBufferMemory),
    "Failed to  map vertex buffer memory");
  auto dataToCopy = std::span{
    reinterpret_cast<const std::byte*>(c_vertices.data()),
    bufferSize
  };
  std::copy(dataToCopy.begin(), dataToCopy.end(), pBufferMemory);
  m_device.unmapMemory(stagingBufferMemory);

  // Create device-local buffer
  auto [buffer, bufferMemory] =
    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferDst |
                 vk::BufferUsageFlagBits::eVertexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal);
  m_vertexBuffer = buffer;
  m_vertexBufferMemory = bufferMemory;

  // Copy data between buffers
  copyBuffer(stagingBuffer, m_vertexBuffer, bufferSize);

  // Clean up the staging buffer
  m_device.destroyBuffer(stagingBuffer);
  m_device.freeMemory(stagingBufferMemory);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createIndexBuffer() {

  // Create staging buffer
  constexpr vk::DeviceSize bufferSize = sizeof(c_indices[0]) * c_indices.size();
  vk::MemoryPropertyFlags memFlags = vk::MemoryPropertyFlagBits::eHostVisible |
                                     vk::MemoryPropertyFlagBits::eHostCoherent;

  auto [stagingBuffer, stagingBufferMemory] =
    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent);

  // Copy data to staging buffer
  std::byte* pBufferMemory{nullptr};
  void** ppBufferMemory = reinterpret_cast<void**>(&pBufferMemory);
  throwOnFailure(
    m_device.mapMemory(stagingBufferMemory, 0, bufferSize, {}, ppBufferMemory),
    "Failed to  map index buffer memory");
  auto dataToCopy = std::span{
    reinterpret_cast<const std::byte*>(c_indices.data()),
    bufferSize
  };
  std::copy(dataToCopy.begin(), dataToCopy.end(), pBufferMemory);
  m_device.unmapMemory(stagingBufferMemory);

  // Create device-local buffer
  auto [buffer, bufferMemory] =
    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferDst |
                 vk::BufferUsageFlagBits::eIndexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal);
  m_indexBuffer = buffer;
  m_indexBufferMemory = bufferMemory;

  // Copy data between buffers
  copyBuffer(stagingBuffer, m_indexBuffer, bufferSize);

  // Clean up the staging buffer
  m_device.destroyBuffer(stagingBuffer);
  m_device.freeMemory(stagingBufferMemory);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);
  vk::CommandPoolCreateInfo poolInfo = {
    .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
  };
  m_commandPool = m_device.createCommandPool(poolInfo);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createCommandBuffers() {
  // Allocate
  vk::CommandBufferAllocateInfo allocInfo = {
    .commandPool = m_commandPool,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = static_cast<uint32_t>(m_swapChainFramebuffers.size())
  };
  m_commandBuffers = m_device.allocateCommandBuffers(allocInfo);

  // Fill
  size_t iBuffer{0};
  for (vk::CommandBuffer buffer : m_commandBuffers) {
    buffer.begin(vk::CommandBufferBeginInfo{});
    vk::ClearValue clearColor = vk::ClearColorValue(std::array{0.0F, 0.0F, 0.0F, 1.0F});
    vk::RenderPassBeginInfo renderPassInfo = {
      .renderPass = m_renderPass,
      .framebuffer = m_swapChainFramebuffers[iBuffer],
      .renderArea = vk::Rect2D{
        .offset = {0, 0},
        .extent = m_swapChainExtent
      },
      .clearValueCount = 1,
      .pClearValues = &clearColor
    };
    buffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);
    buffer.bindVertexBuffers(0, {m_vertexBuffer}, {0});
    buffer.bindIndexBuffer(m_indexBuffer, 0, vk::IndexType::eUint16);
    buffer.drawIndexed(c_indices.size(), 1, 0, 0, 0);
    buffer.endRenderPass();
    buffer.end();
    ++iBuffer;
  }
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::createSyncObjects() {
  vk::FenceCreateInfo fenceInfo = {
    .flags = vk::FenceCreateFlagBits::eSignaled
  };
  for (size_t i=0; i<c_maxFramesInFlight; ++i) {
    m_imageAvailableSemaphores.push_back(
      m_device.createSemaphore(vk::SemaphoreCreateInfo{}));
    m_renderFinishedSemaphores.push_back(
      m_device.createSemaphore(vk::SemaphoreCreateInfo{}));
    m_inFlightFences.push_back(m_device.createFence(fenceInfo));
  }
  m_imagesInFlight.resize(m_swapChainImages.size(), nullptr);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::cleanupSwapChain() {
  for (vk::Framebuffer framebuffer : m_swapChainFramebuffers) {
    m_device.destroyFramebuffer(framebuffer);
  }
  m_swapChainFramebuffers.clear();

  m_device.freeCommandBuffers(m_commandPool, m_commandBuffers);

  m_device.destroyPipeline(m_graphicsPipeline);
  m_device.destroyPipelineLayout(m_pipelineLayout);
  m_device.destroyRenderPass(m_renderPass);

  for (vk::ImageView view: m_swapChainImageViews) {
    m_device.destroyImageView(view);
  }
  m_swapChainImageViews.clear();
  m_device.destroySwapchainKHR(m_swapChain);
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::recreateSwapChain() {
  int width{0};
  int height{0};
  glfwGetFramebufferSize(m_window, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(m_window, &width, &height);
    glfwWaitEvents();
  }

  m_device.waitIdle();

  cleanupSwapChain();

  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandBuffers();
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::drawFrame() {
  static FpsCounter fpsCounter{c_fpsCounterInterval};
  ++fpsCounter;

  static size_t currentFrame{0};
  throwOnFailure(m_device.waitForFences(
    1,
    &m_inFlightFences[currentFrame],
    VK_TRUE,
    std::numeric_limits<uint64_t>::max()));

  vk::ResultValue<uint32_t> nextImageResult = m_device.acquireNextImageKHR(
    m_swapChain,
    std::numeric_limits<uint64_t>::max(),
    m_imageAvailableSemaphores[currentFrame]
  );
  if (nextImageResult.result == vk::Result::eErrorOutOfDateKHR) {
    recreateSwapChain();
    return;
  }
  if (nextImageResult.result != vk::Result::eSuboptimalKHR) {
    throwOnFailure(nextImageResult.result);
  }

  uint32_t imageIndex = nextImageResult.value;

  // Check if a previous frame is using this image (i.e. there is its fence to wait on)
  if (m_imagesInFlight[imageIndex] != vk::Fence{nullptr}) {
    throwOnFailure(m_device.waitForFences(
      1,
      &m_imagesInFlight[imageIndex],
      VK_TRUE,
      std::numeric_limits<uint64_t>::max()));
  }
  // Mark the image as now being in use by this frame
  m_imagesInFlight[imageIndex] = m_inFlightFences[currentFrame];

  // Submit draw command
  vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  vk::SubmitInfo submitInfo = {
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &m_imageAvailableSemaphores[currentFrame],
    .pWaitDstStageMask = &waitStage,
    .commandBufferCount = 1,
    .pCommandBuffers = &m_commandBuffers[imageIndex],
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &m_renderFinishedSemaphores[currentFrame]
  };
  throwOnFailure(m_device.resetFences(1, &m_inFlightFences[currentFrame]));
  throwOnFailure(m_graphicsQueue.submit(1, &submitInfo, m_inFlightFences[currentFrame]),
                 "Failed to submit draw command buffer");

  // Submit present command
  vk::PresentInfoKHR presentInfo = {
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &m_renderFinishedSemaphores[currentFrame],
    .swapchainCount = 1,
    .pSwapchains = &m_swapChain,
    .pImageIndices = &imageIndex
  };
  vk::Result presentResult = m_presentQueue.presentKHR(&presentInfo);
  if (presentResult == vk::Result::eErrorOutOfDateKHR ||
      presentResult == vk::Result::eSuboptimalKHR ||
      m_framebufferResized) {
    m_framebufferResized = false;
    recreateSwapChain();
  }
  else {
    throwOnFailure(presentResult, "Failed to present the image");
  }

  currentFrame = (currentFrame + 1) % c_maxFramesInFlight;
}

// -----------------------------------------------------------------------------
vk::ShaderModule HelloTriangleApplication::createShaderModule(std::vector<char> code) {
  vk::ShaderModuleCreateInfo createInfo = {
    .codeSize = code.size(),
    .pCode = reinterpret_cast<const uint32_t*>(code.data())
  };
  return m_device.createShaderModule(createInfo);
}

// -----------------------------------------------------------------------------
HelloTriangleApplication::QueueFamilyIndices
HelloTriangleApplication::findQueueFamilies(vk::PhysicalDevice device) {
  QueueFamilyIndices indices{};
  std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
  std::cout << "found " << queueFamilies.size() << " queue families" << std::endl;
  auto hasGraphicsBit = [](vk::QueueFamilyProperties fam) -> bool {
    return static_cast<bool>(fam.queueFlags & vk::QueueFlagBits::eGraphics);
  };
  auto hasSurfaceSupport = [&device, this](uint32_t famIndex) -> bool {
    vk::Bool32 supported{VK_FALSE};
    vk::Result result = device.getSurfaceSupportKHR(famIndex, m_surface, &supported);
    return result == vk::Result::eSuccess && supported == VK_TRUE;
  };

  for (size_t famIndex=0; famIndex<queueFamilies.size(); ++famIndex) {
    if (!indices.graphicsFamily.has_value() &&
        hasGraphicsBit(queueFamilies[famIndex])) {
      indices.graphicsFamily = famIndex;
      std::cout << "\tfamily " << famIndex << " supports drawing" << std::endl;
    }
    if (!indices.presentFamily.has_value() &&
        hasSurfaceSupport(famIndex)) {
      indices.presentFamily = famIndex;
      std::cout << "\tfamily " << famIndex << " supports presentation" << std::endl;
    }
    if (indices.isComplete()) {break;}
  }

  return indices;
}

// -----------------------------------------------------------------------------
bool HelloTriangleApplication::isDeviceSuitable(vk::PhysicalDevice device) {
  /*
  vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
  vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();
  if (deviceProperties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu) {return false;}
  if (deviceFeatures.geometryShader == VK_FALSE) {return false;}
  */
  QueueFamilyIndices indices = findQueueFamilies(device);
  if (not indices.isComplete()) {return false;}

  bool extensionsSupported = checkDeviceExtensionSupport(device);
  if (not extensionsSupported) {return false;}

  SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
  if (swapChainSupport.formats.empty()) {return false;}
  if (swapChainSupport.presentModes.empty()) {return false;}

  return true;
}

// -----------------------------------------------------------------------------
HelloTriangleApplication::SwapChainSupportDetails
HelloTriangleApplication::querySwapChainSupport(vk::PhysicalDevice device) {
  SwapChainSupportDetails details{
    .capabilities = device.getSurfaceCapabilitiesKHR(m_surface),
    .formats = device.getSurfaceFormatsKHR(m_surface),
    .presentModes = device.getSurfacePresentModesKHR(m_surface)
  };
  return details;
}

// -----------------------------------------------------------------------------
vk::Extent2D
HelloTriangleApplication::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }

  int frameBufferWidth{0};
  int frameBufferHeight{0};
  glfwGetFramebufferSize(m_window, &frameBufferWidth, &frameBufferHeight);

  return vk::Extent2D{
    .width = std::max(capabilities.minImageExtent.width,
                      std::min(capabilities.maxImageExtent.width,
                               static_cast<uint32_t>(frameBufferWidth))),
    .height = std::max(capabilities.minImageExtent.height,
                       std::min(capabilities.maxImageExtent.height,
                                static_cast<uint32_t>(frameBufferHeight)))
  };
}

// -----------------------------------------------------------------------------
uint32_t HelloTriangleApplication::findMemoryType(
  uint32_t typeFilter, vk::MemoryPropertyFlags properties) {

  vk::PhysicalDeviceMemoryProperties memProperties = m_physicalDevice.getMemoryProperties();
  for (uint32_t i=0; i<memProperties.memoryTypeCount; ++i) {
    if ((typeFilter & (1 << i)) != 0 &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throwOnFailure(false, "Failed to find suitable memory type");
  return std::numeric_limits<uint32_t>::max();
}

// -----------------------------------------------------------------------------
std::pair<vk::Buffer, vk::DeviceMemory> HelloTriangleApplication::createBuffer(
  vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {

  vk::BufferCreateInfo bufferInfo = {
    .size = size,
    .usage = usage,
    .sharingMode = vk::SharingMode::eExclusive
  };
  vk::Buffer buffer = m_device.createBuffer(bufferInfo);

  vk::MemoryRequirements memRequirements =
    m_device.getBufferMemoryRequirements(buffer);

  vk::MemoryAllocateInfo allocInfo = {
    .allocationSize = memRequirements.size,
    .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
  };

  vk::DeviceMemory bufferMemory = m_device.allocateMemory(allocInfo);
  m_device.bindBufferMemory(buffer, bufferMemory, 0);

  return {buffer, bufferMemory};
}

// -----------------------------------------------------------------------------
void  HelloTriangleApplication::copyBuffer(
  vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {

  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);
  vk::CommandPoolCreateInfo poolInfo = {
    .flags = vk::CommandPoolCreateFlagBits::eTransient,
    .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
  };
  vk::CommandPool transientCommandPool = m_device.createCommandPool(poolInfo);

  vk::CommandBufferAllocateInfo allocInfo = {
    .commandPool = transientCommandPool,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = 1
  };
  vk::CommandBuffer copyCommandBuffer = m_device.allocateCommandBuffers(allocInfo).at(0);

  vk::CommandBufferBeginInfo beginInfo = {
    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
  };
  copyCommandBuffer.begin(beginInfo);

  vk::BufferCopy copyRegion = {
    .size = size
  };
  copyCommandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
  copyCommandBuffer.end();

  vk::SubmitInfo submitInfo = {
    .commandBufferCount = 1,
    .pCommandBuffers = &copyCommandBuffer
  };
  throwOnFailure(m_graphicsQueue.submit(1, &submitInfo, nullptr),
                 "Failed to submit copy command buffer");
  m_graphicsQueue.waitIdle();

  m_device.freeCommandBuffers(transientCommandPool, copyCommandBuffer);
  m_device.destroyCommandPool(transientCommandPool);
}

// -----------------------------------------------------------------------------
bool HelloTriangleApplication::checkValidationLayerSupport() {
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

// -----------------------------------------------------------------------------
bool HelloTriangleApplication::checkDeviceExtensionSupport(vk::PhysicalDevice device) {
  size_t notFound{0};
  std::vector<vk::ExtensionProperties> availableExtensions =
    device.enumerateDeviceExtensionProperties();
  std::cout << "required device extensions:" << std::endl;
  for (const auto& extName : c_deviceExtensions) {
    std::cout << '\t' << extName << std::endl;
  }
  std::cout << "available device extensions:" << std::endl;
  for (const auto& ext : availableExtensions) {
    std::cout << '\t' << ext.extensionName << std::endl;
  }
  for (const auto& extName : std::span{c_deviceExtensions}) {
    auto matchName = [&extName](const vk::ExtensionProperties& e){
      return std::string_view(e.extensionName) == std::string_view(extName);
    };
    if (std::ranges::find_if(availableExtensions, matchName) == availableExtensions.end()) {
      std::cout << "Device extension " << extName << " is not available" << std::endl;
      ++notFound;
    }
  }
  return notFound == 0;
}

// -----------------------------------------------------------------------------
std::vector<const char*> HelloTriangleApplication::getRequiredExtensions() {
  // ---------------------------------------------
  // Build the list of required extensions
  // ---------------------------------------------
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

  // ---------------------------------------------
  // Check if the extensions are available
  // ---------------------------------------------
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

// -----------------------------------------------------------------------------
std::vector<char> HelloTriangleApplication::readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  throwOnFailure(file.is_open(), "Failed to open file");
  const size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  std::cout << "Loaded " << buffer.size() << " bytes from file " << filename << std::endl;
  return buffer;
}

// -----------------------------------------------------------------------------
vk::SurfaceFormatKHR HelloTriangleApplication::chooseSwapSurfaceFormat(
  const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
  auto selected = std::ranges::find_if(availableFormats, [](vk::SurfaceFormatKHR fmt){
    return fmt.format  == vk::Format::eB8G8R8A8Srgb &&
           fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
  });
  if (selected != availableFormats.end()) {
    return *selected;
  }
  return availableFormats.at(0);
}

// -----------------------------------------------------------------------------
vk::PresentModeKHR HelloTriangleApplication::chooseSwapPresentMode(
  const std::vector<vk::PresentModeKHR>& availablePresentModes) {
  auto selected = std::ranges::find(availablePresentModes, vk::PresentModeKHR::eMailbox);
  if (selected != availablePresentModes.end()) {
    return *selected;
  }
  return vk::PresentModeKHR::eFifo;
}

// -----------------------------------------------------------------------------
void HelloTriangleApplication::framebufferResizeCallback(
  GLFWwindow* window, int /*width*/, int /*height*/) {
  auto* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
  app->setFramebufferResized(true);
}

// -----------------------------------------------------------------------------
vk::DebugUtilsMessengerCreateInfoEXT HelloTriangleApplication::createDebugMessengerCreateInfo() {
  using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
  using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
  return {
    .messageSeverity = /*Severity::eVerbose |*/ Severity::eInfo |
                       Severity::eWarning | Severity::eError,
    .messageType = Type::eGeneral | Type::eValidation | Type::ePerformance,
    .pfnUserCallback = debugCallback,
    .pUserData = nullptr
  };
}

// -----------------------------------------------------------------------------
VKAPI_ATTR VkBool32 VKAPI_CALL HelloTriangleApplication::debugCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
  VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  void* /*pUserData*/) {

  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

// -----------------------------------------------------------------------------
FpsCounter& FpsCounter::operator++() {
  if (++m_frameNumber == m_maxFrameNumber) {
    auto newTimestamp = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<float>(newTimestamp - m_lastTimestamp);
    float fps = static_cast<float>(m_maxFrameNumber) / duration.count();
    std::cout << "FPS: " << fps << std::endl;
    m_frameNumber = 0;
    m_lastTimestamp = newTimestamp;
  }
  return *this;
}

// -----------------------------------------------------------------------------
vk::VertexInputBindingDescription Vertex::getBindingDescription() {
  return vk::VertexInputBindingDescription{
    .binding = 0,
    .stride = sizeof(Vertex),
    .inputRate = vk::VertexInputRate::eVertex
  };
}

// -----------------------------------------------------------------------------
std::array<vk::VertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions() {
  return std::array{
    vk::VertexInputAttributeDescription{
      .location = 0,
      .binding = 0,
      .format = vk::Format::eR32G32Sfloat,
      .offset = offsetof(Vertex, pos)
    },
    vk::VertexInputAttributeDescription{
      .location = 1,
      .binding = 0,
      .format = vk::Format::eR32G32B32Sfloat,
      .offset = offsetof(Vertex, colour)
    },
  };
}
