# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
SET(PROJECT_NAME dosecalc)
project(${PROJECT_NAME})

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
#
include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(${PROJECT_NAME} main.cc ${sources} ${headers})
target_link_libraries(${PROJECT_NAME} ${Geant4_LIBRARIES} -lprotobuf)
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

#----------------------------------------------------------------------------
# Install the executable to CMAKE_INSTALL_PREFIX with supporing files
file(GLOB input_files
    ${PROJECT_SOURCE_DIR}/inputs/*.in
    ${PROJECT_SOURCE_DIR}/inputs/*.mac
    # ${PROJECT_SOURCE_DIR}/utils/bulk_simulation*.py
)
install(FILES ${input_files} DESTINATION inputs/.)

install(TARGETS ${PROJECT_NAME} DESTINATION bin/.)
