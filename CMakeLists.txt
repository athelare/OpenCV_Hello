cmake_minimum_required(VERSION 3.15)
project(OpenCV_Hello)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS"-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG"-g")
set(CMAKE_CXX_FLAGS_RELEASE"-O3")
set(CMAKE_CXX_FLAGS_DEBUG"-O3")
set(OpenCV_DIR D:/OpenCV/opencv/build)
INCLUDE_DIRECTORIES(D:\\curl-7.76.1\\builds\\libcurl-vc15-x64-debug-static-ipv6-sspi-schannel\\include)
LINK_DIRECTORIES(D:/curl-7.76.1/builds/libcurl-vc15-x64-debug-static-ipv6-sspi-schannel/lib/)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_executable(OpenCV_Hello main.cpp ../cv_target_vision/src/utils.cpp Dvp2Getframe.cpp getPhotoFromMultipleCameras.cpp DvpFrameCapture.h DetectStraitOrNot.cpp SmoothContourTest.cpp main.h CannyTest.cpp SendImage.cpp)
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(OpenCV_Hello ${OpenCV_LIBS} libcurl_a_debug ws2_32 wldap32 Crypt32 Normaliz D:\\Source\\CLionProjects\\OpenCV_Hello\\lib\\x64\\DVPCamera64.lib)
ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)

