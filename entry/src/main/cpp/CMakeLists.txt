cmake_minimum_required(VERSION 3.5.0)
project(voicehealth)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

add_library(voicehealth SHARED
        mel_frontend.cpp
        attrib.cpp
        napi_init.cpp)

# libmindspore_lite_ndk.so ships in the OHOS NDK sysroot; the system provides it at runtime.
target_link_libraries(voicehealth PUBLIC
        libace_napi.z.so
        libhilog_ndk.z.so
        mindspore_lite_ndk)
