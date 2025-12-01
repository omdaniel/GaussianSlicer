cmake_minimum_required(VERSION 3.15)
project(vdb_writer)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenVDB via config mode (provided by vcpkg)
find_package(OpenVDB CONFIG REQUIRED)

# Add executable
add_executable(vdb_writer vdb_writer.cpp)

# Link OpenVDB
target_link_libraries(vdb_writer PRIVATE OpenVDB::openvdb)

# Copy to output directory if needed, or just leave it in build dir
