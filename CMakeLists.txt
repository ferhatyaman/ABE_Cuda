cmake_minimum_required (VERSION 3.5.1)
project(ABE_Cuda C CUDA)

set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CXX_STANDARD 5)

find_package(Palisade)

#set( CMAKE_CXX_FLAGS ${PALISADE_CXX_FLAGS} )
#set( CMAKE_EXE_LINKER_FLAGS ${PALISADE_EXE_LINKER_FLAGS} )

set( CMAKE_NVCC_FLAGS ${PALISADE_CUDA_FLAGS})

#[[include_directories( /home/usr/local/include/palisade/core )]]
include_directories(${CORE_INCLUDE_DIRS}) 
list(APPEND PKE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include") 
list(APPEND PKE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/lib") 
include_directories( /home/usr/local/include/palisade/core ) 

include_directories(${PKE_INCLUDE_DIRS})
#include_directories( /home/usr/local/include/palisade/pke )
#include_directories( /home/usr/local/include)

include_directories( ${OPENMP_INCLUDES})
include_directories( ${PALISADE_INCLUDE})
include_directories( ${PALISADE_INCLUDE}/math)
#include_directories( ${PALISADE_INCLUDE}/include )
include_directories( ${PALISADE_INCLUDE}/third-party/include )
include_directories( ${PALISADE_INCLUDE}/core)
include_directories( ${PALISADE_INCLUDE}/core/math)
include_directories( ${PALISADE_INCLUDE}/pke )
include_directories( ${PALISADE_INCLUDE}/cereal )
#include_directories( ${PALISADE_INCLUDE}/abe )
#include_directories( ${PALISADE_INCLUDE}/trapdoor )
#include_directories( ${PALISADE_INCLUDE}/signature )

### add directories for other PALISADE modules as needed for your project

link_directories( ${PALISADE_LIBDIR})
link_directories( ${OPENMP_LIBRARIES})
link_libraries( ${PALISADE_LIBRARIES})


#add_executable(deneme deneme.cu)

add_executable(ABE_Cuda add-ckks.cu)
set_target_properties(ABE_Cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
#
#set_target_properties(
#        deneme
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)




### To use gcc/g++ on a Macintosh, you must set the Compilers
### here, not inside the project
##if(APPLE)
##       set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
##       set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
##endif()
### TODO: for now, we use CLang for Mac






### ADD YOUR EXECUTABLE(s) HERE
### add_executable( EXECUTABLE-NAME SOURCES )
###
### EXAMPLE:
### add_executable( test demo-simple-example.cpp )
