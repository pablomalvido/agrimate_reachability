cmake_minimum_required(VERSION 3.8)
project(moveit_cpp_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(custom_interfaces REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(move_robot src/move_robot.cpp)
add_executable(ik_checker src/reachability_ik_checker_v3.cpp)
add_executable(ik_test src/ik_test.cpp)
add_executable(collisions_server src/collisions_server.cpp)
add_executable(ik_collisions src/ik_collisions_v2.cpp)


ament_target_dependencies(move_robot 
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
)

ament_target_dependencies(ik_checker 
  rclcpp
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

ament_target_dependencies(ik_test 
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
)

ament_target_dependencies(collisions_server 
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  moveit_msgs
  shape_msgs
  custom_interfaces
)

ament_target_dependencies(ik_collisions 
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  moveit_msgs
  shape_msgs
)

install(TARGETS move_robot ik_checker ik_test collisions_server ik_collisions
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
