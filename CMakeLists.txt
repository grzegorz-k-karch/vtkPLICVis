CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(COMMAND cmake_policy)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND cmake_policy)

IF (NOT ParaView_BINARY_DIR)
  FIND_PACKAGE(ParaView REQUIRED)
  INCLUDE(${PARAVIEW_USE_FILE})
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -arch compute_30")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)

CUDA_ADD_LIBRARY(PLICVis_impl
  SHARED
  plicvis_impl.cu)

ADD_PARAVIEW_PLUGIN(PLICVis "1.0"
  SERVER_MANAGER_XML PLICVis.xml
  SERVER_MANAGER_SOURCES vtkPLICVis.cxx)

TARGET_LINK_LIBRARIES(PLICVis PRIVATE PLICVis_impl)
