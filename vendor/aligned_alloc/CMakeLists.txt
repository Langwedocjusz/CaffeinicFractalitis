cmake_minimum_required(VERSION 3.16.3)

project(aligned_alloc)

add_library(aligned_alloc INTERFACE)
target_include_directories(aligned_alloc SYSTEM INTERFACE .)