#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <memory>
#include <string>
#include <thread>
#include <chrono>

class IKReachabilityNode : public rclcpp::Node
{
public:
  IKReachabilityNode()
  : Node("ik_reachability_checker"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    RCLCPP_INFO(this->get_logger(), "Node created");
  }

  void initialize()
  {
    RCLCPP_INFO(this->get_logger(), "Initializing IK checker...");

    // Small delay to allow TF + robot_description to be available
    rclcpp::sleep_for(std::chrono::seconds(2));

    // Load robot model (SAFE now)
    robot_model_loader::RobotModelLoader model_loader(
      shared_from_this(), "robot_description");

    robot_model_ = model_loader.getModel();

    if (!robot_model_) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load robot model!");
      return;
    }

    robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    robot_state_->setToDefaultValues();

    joint_model_group_ = robot_model_->getJointModelGroup("ur_manipulator");

    if (!joint_model_group_) {
      RCLCPP_ERROR(this->get_logger(), "Joint model group not found!");
      return;
    }

    base_frame_ = robot_model_->getModelFrame();

    RCLCPP_INFO(this->get_logger(), "Base frame: %s", base_frame_.c_str());


    checkReachability();
  }

private:
  void checkReachability()
  {
    int success_count = 0;
    int total = 50;

    RCLCPP_INFO(this->get_logger(), "Checking %d poses...", total);

    for (int i = 0; i < total; ++i)
    {
      std::string frame_name = "sample_pose_" + std::to_string(i);

      geometry_msgs::msg::TransformStamped transform;

      try
      {
        transform = tf_buffer_.lookupTransform(
          base_frame_, frame_name, tf2::TimePointZero);
      }
      catch (tf2::TransformException &ex)
      {
        RCLCPP_WARN(this->get_logger(),
                    "TF not found for %s: %s",
                    frame_name.c_str(), ex.what());
        continue;
      }

      // Convert TF to Pose
      geometry_msgs::msg::Pose pose;
      pose.position.x = transform.transform.translation.x;
      pose.position.y = transform.transform.translation.y;
      pose.position.z = transform.transform.translation.z;
      pose.orientation = transform.transform.rotation;

      RCLCPP_INFO(this->get_logger(), "POSE: %f, %f, %f, %f, %f, %f, %f", pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);

      // Reset state before IK
      robot_state_->setToDefaultValues();

      // Try IK
      bool found_ik = robot_state_->setFromIK(
        joint_model_group_,
        pose,
        0.1  // timeout (seconds)
      );

      if (found_ik)
      {
        success_count++;
        RCLCPP_INFO(this->get_logger(), "IK success: %s", frame_name.c_str());
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "IK failed: %s", frame_name.c_str());
      }
    }

    RCLCPP_INFO(this->get_logger(),
                "Reachable poses: %d / %d",
                success_count, total);
  }

  // Members
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  const moveit::core::JointModelGroup* joint_model_group_;

  std::string base_frame_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<IKReachabilityNode>();

  // Run initialization in a separate thread so spin() can run
  std::thread init_thread([node]() {
    node->initialize();
  });

  rclcpp::spin(node);

  init_thread.join();

  rclcpp::shutdown();
  return 0;
}