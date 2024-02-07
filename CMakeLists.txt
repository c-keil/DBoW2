cmake_minimum_required(VERSION 3.0)
project(DBoW2)
include(ExternalProject)

option(BUILD_DBoW2   "Build DBoW2"            ON)
option(BUILD_Demo    "Build demo application" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# # for cnpy
find_library(LIBCNPY libcnpy.so)
message(LIBCNPY="${LIBCNPY}")

set(HDRS
  include/DBoW2/BowVector.h           
  include/DBoW2/FBrief.h
  include/DBoW2/QueryResults.h        
  include/DBoW2/TemplatedDatabase.h   
  include/DBoW2/FORB.h
  include/DBoW2/DBoW2.h               
  include/DBoW2/FClass.h              
  include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       
  include/DBoW2/TemplatedVocabulary.h
  include/DBoW2/FIR.h
  include/DBoW2/FIR2.h
  include/DBoW2/read_ir.h)
set(SRCS 
  src/BowVector.cpp     
  src/FBrief.cpp        
  src/FORB.cpp
  src/FIR.cpp
  src/FIR2.cpp
  src/FeatureVector.cpp 
  src/QueryResults.cpp  
  src/ScoringObject.cpp
  src/read_ir.cpp)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)


# if(DEFINED ENV{OPENCV_INSTALL_DIR})
#     find_package(OpenCV 3.2 REQUIRED PATHS $ENV{OPENCV_INSTALL_DIR})
# else()
#     message("OPENCV_INSTALL_DIR not set, searching in default location(s)")
#     # find_package(OpenCV 3.2 REQUIRED)
#     find_package(OpenCV REQUIRED)
# endif(DEFINED ENV{OPENCV_INSTALL_DIR})

find_package(OpenCV REQUIRED)
# find_package(OpenCV 4.4 REQUIRED)
# find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(BUILD_DBoW2)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
  target_include_directories(${PROJECT_NAME} PUBLIC include/DBoW2/ include/)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY}) #mod for libcnpy
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
endif(BUILD_DBoW2)

if(BUILD_Demo)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 11)
  file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)

  add_executable(ir_demo demo/ir_demo.cpp)
  target_link_libraries(ir_demo ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
  set_target_properties(ir_demo PROPERTIES CXX_STANDARD 11)
  add_executable(ir_demo2 demo/ir_demo2.cpp)
  target_link_libraries(ir_demo2 ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
  set_target_properties(ir_demo2 PROPERTIES CXX_STANDARD 11)
  # file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)
  endif(BUILD_Demo)
#IR experiment
add_executable(ir_timing_exp scripts/ir_timing_exp.cpp)
target_link_libraries(ir_timing_exp ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
set_target_properties(ir_timing_exp PROPERTIES CXX_STANDARD 11)

add_executable(ir_debug demo/ir_debug.cpp)
target_link_libraries(ir_debug ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
set_target_properties(ir_debug PROPERTIES CXX_STANDARD 11)

add_executable(build_ir_voc scripts/build_ir_voc.cpp)
target_link_libraries(build_ir_voc ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
set_target_properties(build_ir_voc PROPERTIES CXX_STANDARD 11)

add_executable(corss_test_voc scripts/test_ir_voc.cpp)
target_link_libraries(corss_test_voc ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
set_target_properties(corss_test_voc PROPERTIES CXX_STANDARD 11)

add_executable(test_ir_voc_tg scripts/test_ir_temporal_gap_voc.cpp)
target_link_libraries(test_ir_voc_tg ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
set_target_properties(test_ir_voc_tg PROPERTIES CXX_STANDARD 11)

add_executable(simple_ir_test scripts/simple_ir_test.cpp)
target_link_libraries(simple_ir_test ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
set_target_properties(simple_ir_test PROPERTIES CXX_STANDARD 11)

add_executable(ir_db_experiment scripts/ir_db_experiment.cpp)
target_link_libraries(ir_db_experiment ${PROJECT_NAME} ${OpenCV_LIBS} ${LIBCNPY})
set_target_properties(ir_db_experiment PROPERTIES CXX_STANDARD 11)

configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DBoW2)
  install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)
message(CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
message(PROJECT_BINARY_DIR="${PROJECT_BINARY_DIR}")

