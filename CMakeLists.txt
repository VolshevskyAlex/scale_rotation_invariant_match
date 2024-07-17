project(scale_rotation_invariant_match)
#set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -std=c++17")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(scale_rotation_invariant_match
main2.cpp
match_pict.cpp
split_to_tiles.cpp
)

target_link_libraries(scale_rotation_invariant_match
${OpenCV_LIBS}
)
