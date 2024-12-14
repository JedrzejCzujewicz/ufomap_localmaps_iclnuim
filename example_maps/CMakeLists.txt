################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(example_maps)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package()

################################################################################
# Build
################################################################################
include_directories(include ${catkin_INCLUDE_DIRS}) 

################################################################################
# Install
################################################################################
install(DIRECTORY launch maps param rviz 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/map_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
################################################################################
# Test
################################################################################
