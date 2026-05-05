#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>

#include <moveit/collision_detection/collision_common.h>

#include <geometry_msgs/msg/pose.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("ik_only_example");

  // Spin so parameters are available
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  // Load robot model
  robot_model_loader::RobotModelLoader loader(node, "robot_description");
  auto robot_model = loader.getModel();

  if (!robot_model)
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to load robot model");
    return 1;
  }

  // Robot state
  moveit::core::RobotState robot_state(robot_model);
  robot_state.setToDefaultValues();

  // Joint group
  const moveit::core::JointModelGroup* jmg =
    robot_model->getJointModelGroup("ur_manipulator");

  if (!jmg)
  {
    RCLCPP_ERROR(node->get_logger(), "Joint group not found");
    return 1;
  }

  // Planning scene
  auto planning_scene =
    std::make_shared<planning_scene::PlanningScene>(robot_model);

  // Target pose
  geometry_msgs::msg::Pose target_pose;
  target_pose.position.x = 0.26554;
  target_pose.position.y = -0.81087;
  target_pose.position.z = 1.1668;
  target_pose.orientation.x = -0.632248;
  target_pose.orientation.y = 0.663155;
  target_pose.orientation.z = -0.214152;
  target_pose.orientation.w = -0.338566;

  // Solve IK
  bool found_ik = robot_state.setFromIK(
    jmg,
    target_pose,
    0.1
  );

  if (found_ik)
  {
    RCLCPP_INFO(node->get_logger(), "IK solution found!");

    std::vector<double> joint_values;
    robot_state.copyJointGroupPositions(jmg, joint_values);

    for (size_t i = 0; i < joint_values.size(); ++i)
    {
      RCLCPP_INFO(node->get_logger(),
                  "Joint %zu: %f", i, joint_values[i]);
    }

    // Collision check
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;

    planning_scene->checkCollision(req, res, robot_state);

    if (res.collision)
    {
      RCLCPP_WARN(node->get_logger(), "IK solution is in collision!");
    }
    else
    {
      RCLCPP_INFO(node->get_logger(), "IK solution is collision-free!");
    }
  }
  else
  {
    RCLCPP_WARN(node->get_logger(), "IK failed");
  }

  rclcpp::shutdown();
  return 0;
}