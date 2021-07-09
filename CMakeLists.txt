cmake_minimum_required(VERSION 3.0.2)
project(mu_auv_localization)

find_package(catkin REQUIRED COMPONENTS
  apriltag_ros
  geometry_msgs
  roscpp
  rospy
  std_msgs
)

catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
  nodes/vision_ekf_node
  nodes/vision_estimator_node
  scripts/generate_tag_poses_yaml
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
