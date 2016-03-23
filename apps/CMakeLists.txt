project(GPUFILTERS_APPS)

# Enumerate all subdirs and process them:
list_subdirectories(LIST_GPUFILTERS_APP_DIRS "${GPUFILTERS_SOURCE_DIR}/apps")

foreach(dir ${LIST_GPUFILTERS_APP_DIRS})
	add_subdirectory(${dir})
endforeach(dir)