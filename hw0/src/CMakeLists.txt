cmake_minimum_required(VERSION 3.5)

project(ml)

# 设置 C++ 编译选项
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

add_executable(mytest my_test.cpp)

find_package(GTest REQUIRED)

target_link_libraries(mytest
        GTest::GTest
        GTest::Main
        )