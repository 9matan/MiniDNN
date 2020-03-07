set(TEST2_SOURCES test2.cpp)

if(DEBUG_ADD_MINIDNN_FILES_TO_TESTS)
    set(TEST2_SOURCES ${TEST2_SOURCES} ${MINIDNN_HEADERS})
endif()

add_executable(Test2 ${TEST2_SOURCES})
target_include_directories(Test2 PUBLIC ${MINIDNN_INCLUDE_DIRS})
target_include_directories(Test2 PUBLIC SYSTEM ${MINIDNN_SYSTEM_INCLUDE_DIRS})