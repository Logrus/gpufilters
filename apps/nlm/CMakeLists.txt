project(nlm)

#link_directories(${CMAKE_BINARY_DIR})
#find_library(GPUFILTER_LIBS SHARED nlm_clustertree )

add_executable(nlm nlm.cpp)
# Add the required libraries for linking:
target_link_libraries(nlm  nlm_clustertree nlm_texture nlm_lkop nlm_shared helpers )
