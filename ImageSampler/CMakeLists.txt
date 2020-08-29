cmake_minimum_required(VERSION 3.10)
project(ProjectName)

add_custom_target(resources)
FILE(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/resources" source)
FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/resources" destination)
add_custom_command(
    TARGET resources POST_BUILD
    COMMAND mklink /D ${destination} ${source}
    DEPENDS ${destination}
    COMMENT "symbolic link resources folder from ${source} => ${destination}")

file( GLOB APP_SOURCES *.cpp )
foreach( testsourcefile ${APP_SOURCES} )
    get_filename_component(testname ${testsourcefile} NAME_WLE)
    add_executable( ${testname} ${testsourcefile} )
    set_target_properties(${testname} PROPERTIES CXX_STANDARD 17)
    add_dependencies(${testname} resources)
    
endforeach( testsourcefile ${APP_SOURCES} )