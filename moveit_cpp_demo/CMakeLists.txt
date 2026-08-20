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
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(sensor_msgs REQUIRED)

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
add_executable(ik_checker src/reachability_ik_checker_v5.cpp src/simplified_bayesian_optimizer.cpp src/utils.cpp)
add_executable(ik_checker_3d src/reachability_ik_checker_3d.cpp src/simplified_bayesian_optimizer.cpp src/utils.cpp)
add_executable(ik_test src/ik_test.cpp)
add_executable(collisions_server src/collisions_server.cpp)
add_executable(ik_collisions src/ik_collisions_v2.cpp)
add_executable(fake_state_pub src/fake_joint_publisher.cpp)
add_executable(init_config src/initial_config_optimizer.cpp)
add_executable(init_config3d src/initial_config_optimizer_3d.cpp)


target_include_directories(ik_checker PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(ik_checker_3d PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(init_config PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(init_config3d PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(ik_checker PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_definitions(ik_checker_3d PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

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
  ament_index_cpp
  custom_interfaces
)

ament_target_dependencies(ik_checker_3d
  rclcpp
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  ament_index_cpp
  custom_interfaces
)

ament_target_dependencies(init_config 
  rclcpp
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  ament_index_cpp
  custom_interfaces
)

ament_target_dependencies(init_config3d 
  rclcpp
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  ament_index_cpp
  custom_interfaces
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

ament_target_dependencies(fake_state_pub 
  rclcpp
  sensor_msgs
)

target_link_libraries(ik_checker yaml-cpp)
target_link_libraries(ik_checker_3d yaml-cpp)
target_link_libraries(init_config yaml-cpp)
target_link_libraries(init_config3d yaml-cpp)

install(TARGETS move_robot ik_checker ik_checker_3d init_config init_config3d ik_test collisions_server ik_collisions fake_state_pub
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
