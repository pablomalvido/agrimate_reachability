#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>

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

    // Planning scene
    planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);
    
    joint_model_group_ = robot_model_->getJointModelGroup("ur_manipulator");

    if (!joint_model_group_) {
      RCLCPP_ERROR(this->get_logger(), "Joint model group not found!");
      return;
    }

    base_frame_ = robot_model_->getModelFrame();

    RCLCPP_INFO(this->get_logger(), "Base frame: %s", base_frame_.c_str());


    checkReachability();
  }

  void printCollisionMatrix()
  {
    const collision_detection::AllowedCollisionMatrix& acm =
      planning_scene_->getAllowedCollisionMatrix();

    std::vector<std::string> link_names;
    acm.getAllEntryNames(link_names);

    RCLCPP_INFO(this->get_logger(), "===== Allowed Collision Matrix =====");

    for (size_t i = 0; i < link_names.size(); ++i)
    {
      for (size_t j = i + 1; j < link_names.size(); ++j)
      {
        collision_detection::AllowedCollision::Type type;
        if (acm.getEntry(link_names[i], link_names[j], type))
        {
          if (type != collision_detection::AllowedCollision::NEVER)
          {
            RCLCPP_INFO(this->get_logger(),
              "Allowed: %s <-> %s",
              link_names[i].c_str(),
              link_names[j].c_str());
          }
        }
      }
    }

    RCLCPP_INFO(this->get_logger(), "====================================");
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

    float z_offset = 0.1;
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
  }

  void removeBoxObstacle()
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "temp_box";
    obj.header.frame_id = base_frame_;
    obj.operation = obj.REMOVE;

    planning_scene_->processCollisionObjectMsg(obj);

    // RCLCPP_INFO(get_logger(), "Obstacle removed");
  }

  // =========================
  // COLLISION CHECK
  // =========================
  bool isStateValid(moveit::core::RobotState* state)
  {
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;

    req.contacts = true;
    req.distance = true;

    planning_scene_->checkCollision(req, res, *state);
    
    if (res.collision)
    {
      RCLCPP_WARN(get_logger(), "State in collision");
      for (auto &contact : res.contacts) 
      { 
          RCLCPP_WARN(get_logger(), "Collision: %s <-> %s", contact.first.first.c_str(), contact.first.second.c_str()); 
      } 
    } else {
      RCLCPP_INFO(get_logger(), "Distance to collision: %f", res.distance);
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

  void checkReachability()
  {
    // printCollisionMatrix();

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

      addBoxObstacle(pose);
      bool found_ik = computeIKWithRetries(pose);

      if (found_ik)
      {
        success_count++;
        RCLCPP_INFO(this->get_logger(), "IK success: %s", frame_name.c_str());
      }

      else
      {
        RCLCPP_WARN(this->get_logger(), "IK failed: %s", frame_name.c_str());
      }

      removeBoxObstacle();
    }

    RCLCPP_INFO(this->get_logger(),
                "Reachable poses: %d / %d",
                success_count, total);
  }

  // Members
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  planning_scene::PlanningScenePtr planning_scene_;
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