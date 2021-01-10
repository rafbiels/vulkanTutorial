#include "MyVulkanApp.h"

#include <cstdlib>
#include <iostream>

int main() {
  MyVulkanApp app{};

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
