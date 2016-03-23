project(GPUFILTERS_LIBS)

list_subdirectories(LIST_GPUFILTERS_LIB_DIRS "${GPUFILTERS_SOURCE_DIR}/src")

foreach(dir ${LIST_GPUFILTERS_LIB_DIRS})
  add_subdirectory(${dir})
endforeach(dir)

