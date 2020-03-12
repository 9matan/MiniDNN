set(MNIST_SOURCES mnist.cpp Profiler.cpp Profiler.h)

if(DEBUG_ADD_MINIDNN_FILES_TO_TESTS)
    set(MNIST_SOURCES ${MNIST_SOURCES} ${MINIDNN_HEADERS})
endif()

add_executable(Mnist ${MNIST_SOURCES})
target_compile_definitions(Mnist PUBLIC MNIST_DATA_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_include_directories(Mnist PUBLIC ${MINIDNN_INCLUDE_DIRS})
target_include_directories(Mnist PUBLIC SYSTEM ${MINIDNN_SYSTEM_INCLUDE_DIRS})