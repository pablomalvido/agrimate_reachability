#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>

#include <moveit/collision_detection/collision_common.h>

#include <geometry_msgs/msg/pose.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

#include <memory>
#include <thread>
#include <chrono>

class IKWithDynamicObstacle : public rclcpp::Node
{
public:
  IKWithDynamicObstacle()
  : Node("ik_with_dynamic_obstacle")
  {
    RCLCPP_INFO(get_logger(), "Node created");
  }

  void initialize()
  {
    RCLCPP_INFO(get_logger(), "Initializing...");

    rclcpp::sleep_for(std::chrono::seconds(2));

    robot_model_loader::RobotModelLoader loader(
      shared_from_this(), "robot_description");

    robot_model_ = loader.getModel();

    if (!robot_model_) {
      RCLCPP_ERROR(get_logger(), "Failed to load robot model");
      return;
    }

    planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);

    robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    robot_state_->setToDefaultValues();

    joint_model_group_ = robot_model_->getJointModelGroup("ur_manipulator");

    if (!joint_model_group_) {
      RCLCPP_ERROR(get_logger(), "Joint model group not found!");
      return;
    }

    base_frame_ = robot_model_->getModelFrame();

    planning_scene_pub_ =
      this->create_publisher<moveit_msgs::msg::PlanningScene>(
        "/planning_scene", 10);

    runExample();
  }

private:

  // =========================
  // OBSTACLE
  // =========================
  void addBoxObstacle(const geometry_msgs::msg::Pose& target_pose)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "temp_box";
    obj.header.frame_id = base_frame_;
    obj.pose.position = target_pose.position;
    obj.pose.position.x = 0.15;
    obj.pose.orientation.w = 1.0;

    float z_offset = 0.2;
    float dim_z = target_pose.position.z - z_offset;
    obj.pose.position.z = dim_z/2.0;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {1.0, 0.10, dim_z};

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;

    obj.primitives.push_back(primitive);
    obj.primitive_poses.push_back(box_pose);
    obj.operation = obj.ADD;

    planning_scene_->processCollisionObjectMsg(obj);

    moveit_msgs::msg::PlanningScene ps;
    ps.is_diff = true;
    ps.world.collision_objects.push_back(obj);

    planning_scene_pub_->publish(ps);

    RCLCPP_INFO(get_logger(), "Obstacle added");
  }

  void removeBoxObstacle()
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "temp_box";
    obj.header.frame_id = base_frame_;
    obj.operation = obj.REMOVE;

    planning_scene_->processCollisionObjectMsg(obj);

    RCLCPP_INFO(get_logger(), "Obstacle removed");
  }

  // =========================
  // COLLISION CHECK
  // =========================
  bool isStateValid(moveit::core::RobotState* state)
  {
    /**
    std::vector<double> joint_values;
    state->copyJointGroupPositions(joint_model_group_, joint_values);

    // Get joint names from the group
    const std::vector<std::string>& joint_names = joint_model_group_->getVariableNames();

    for (size_t i = 0; i < joint_values.size(); ++i)
    {
      RCLCPP_INFO(get_logger(),
            "%s: %f",
            joint_names[i].c_str(),
            joint_values[i]);
    }
    **/
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;

    planning_scene_->checkCollision(req, res, *state);

    if (res.collision)
    {
      RCLCPP_WARN(get_logger(), "State in collision");
    }

    return !res.collision;
  }

  // =========================
  // IK WITH RETRIES
  // =========================
  bool computeIKWithRetries(const geometry_msgs::msg::Pose& pose)
  {
    const int max_attempts = 5;

    for (int attempt = 0; attempt < max_attempts; ++attempt)
    {
      RCLCPP_INFO(get_logger(), "IK attempt %d", attempt + 1);

      // Different seed each time
      if (attempt==0)
      {
        robot_state_->setToDefaultValues();
      }
      else{
        robot_state_->setToRandomPositions(joint_model_group_);
      }

      bool found_ik = robot_state_->setFromIK(
        joint_model_group_,
        pose,
        0.05   // solver timeout
      );

      if (!found_ik)
      {
        RCLCPP_WARN(get_logger(), "IK solver failed");
        continue;
      }

      // Check collision AFTER IK
      if (isStateValid(robot_state_.get()))
      {
        RCLCPP_INFO(get_logger(), "Valid solution found!");
        return true;
      }
      else
      {
        RCLCPP_WARN(get_logger(), "Solution in collision, retrying...");
      }
    }

    return false;
  }

  // =========================
  // MAIN LOGIC
  // =========================
  void runExample()
  {
    geometry_msgs::msg::Pose target_pose;

    target_pose.position.x = 0.26554;
    target_pose.position.y = -0.81087;
    target_pose.position.z = 1.1668;
    target_pose.orientation.x = -0.632248;
    target_pose.orientation.y = 0.663155;
    target_pose.orientation.z = -0.214152;
    target_pose.orientation.w = -0.338566;

    RCLCPP_INFO(get_logger(), "Adding obstacle...");
    addBoxObstacle(target_pose);

    bool success = computeIKWithRetries(target_pose);

    if (success)
    {
      RCLCPP_INFO(get_logger(), "IK SUCCESS");

      std::vector<double> joint_values;
      robot_state_->copyJointGroupPositions(joint_model_group_, joint_values);

      const std::vector<std::string>& joint_names =
        joint_model_group_->getVariableNames();

      for (size_t i = 0; i < joint_values.size(); ++i)
      {
        RCLCPP_INFO(get_logger(),
          "%s: %f",
          joint_names[i].c_str(),
          joint_values[i]);
      }
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "No valid IK solution found after retries");
    }

    RCLCPP_INFO(get_logger(), "Removing obstacle...");
    removeBoxObstacle();
  }

  // =========================
  // MEMBERS
  // =========================
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  planning_scene::PlanningScenePtr planning_scene_;
  const moveit::core::JointModelGroup* joint_model_group_;
  std::string base_frame_;
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<IKWithDynamicObstacle>();

  std::thread init_thread([node]() {
    node->initialize();
  });

  rclcpp::spin(node);

  init_thread.join();

  rclcpp::shutdown();
  return 0;
}