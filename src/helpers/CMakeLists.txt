project(helpers)

FILE(GLOB header_libs *.h)

add_library(helpers STATIC  load_sequence.cpp ${header_libs})
