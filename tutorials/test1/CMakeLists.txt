set(TEST1_SOURCES test1.cpp)

if(DEBUG_ADD_MINIDNN_FILES_TO_TESTS)
    set(TEST1_SOURCES ${TEST1_SOURCES} ${MINIDNN_HEADERS})
endif()

add_executable(Test1 ${TEST1_SOURCES})
target_include_directories(Test1 PUBLIC ${MINIDNN_INCLUDE_DIRS})
target_include_directories(Test1 PUBLIC SYSTEM ${MINIDNN_SYSTEM_INCLUDE_DIRS})