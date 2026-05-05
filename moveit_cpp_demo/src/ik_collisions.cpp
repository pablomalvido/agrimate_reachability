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

    // Give ROS time to provide parameters (important!)
    rclcpp::sleep_for(std::chrono::seconds(2));

    // SAFE now
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

  void addBoxObstacle(const geometry_msgs::msg::Pose& target_pose)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "temp_box";
    obj.header.frame_id = base_frame_;
    obj.pose = target_pose;
    obj.pose.position.y -= 0.2;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {0.3, 0.3, 0.3};
    //primitive.dimensions = {0.3, 0.3, 0.3};

    geometry_msgs::msg::Pose box_pose;
    box_pose.position.x = 0.0;
    box_pose.position.y = 0.0;
    box_pose.position.z = 0.0;
    box_pose.orientation.w = 1.0;

    obj.primitives.push_back(primitive);
    obj.primitive_poses.push_back(box_pose);
    obj.operation = obj.ADD;

    planning_scene_->processCollisionObjectMsg(obj);

    moveit_msgs::msg::PlanningScene ps;
    ps.is_diff = true;
    ps.world.collision_objects.push_back(obj);

    planning_scene_pub_->publish(ps);

    RCLCPP_INFO(get_logger(), "Published box to planning_scene");
  }

  void removeBoxObstacle()
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "temp_box";
    obj.header.frame_id = base_frame_;
    obj.operation = obj.REMOVE;

    planning_scene_->processCollisionObjectMsg(obj);
  }

  bool isStateValid(moveit::core::RobotState* state)
  {

    // 1. Collision check
    collision_detection::CollisionRequest col_req;
    collision_detection::CollisionResult col_res;

    col_req.contacts = true;
    col_req.max_contacts = 100;
    col_req.max_contacts_per_pair = 10;
    col_req.distance = true;

    planning_scene_->checkCollision(col_req, col_res, *state);

    if (true) 
    { 
        RCLCPP_INFO(get_logger(), "COLLISION"); 
        //RCLCPP_INFO(get_logger(), "Distance: %f", col_res.distance); 
        for (auto &contact : col_res.contacts) 
        { 
            RCLCPP_WARN(get_logger(), "Collision: %s <-> %s", contact.first.first.c_str(), contact.first.second.c_str()); 
        } 
    }

    // 2. Distance check
    collision_detection::DistanceRequest dist_req;
    collision_detection::DistanceResult dist_res;

    dist_req.enable_nearest_points = true;
    dist_req.enable_signed_distance = true;
    dist_req.compute_gradient = true;   // sometimes needed
    dist_req.type = collision_detection::DistanceRequestType::ALL;

    planning_scene_->getCollisionEnv()->distanceRobot(dist_req, dist_res, *state);

    RCLCPP_INFO(get_logger(), "Min distance: %f", dist_res.minimum_distance.distance);
    RCLCPP_INFO(get_logger(), "Minimum distance links: %s <--> %s", dist_res.minimum_distance.link_names[0].c_str(), dist_res.minimum_distance.link_names[1].c_str()); 
    RCLCPP_INFO(get_logger(), "Minimum distance point 1: %f, %f, %f", dist_res.minimum_distance.nearest_points[0].x(), dist_res.minimum_distance.nearest_points[0].y(), dist_res.minimum_distance.nearest_points[0].z()); 
    RCLCPP_INFO(get_logger(), "Minimum distance point 2: %f, %f, %f", dist_res.minimum_distance.nearest_points[1].x(), dist_res.minimum_distance.nearest_points[1].y(), dist_res.minimum_distance.nearest_points[1].z());

    std::string link1 = "wrist_3_link";  // robot link name
    std::string link2 = "temp_box";  // collision object id
    RCLCPP_INFO(get_logger(),
        "Distance map size: %zu",
        dist_res.distances.size());

    for (const auto& entry : dist_res.distances)
    {
    const auto& pair = entry.first;
    const auto& vec  = entry.second;

    RCLCPP_INFO(get_logger(),
      "Pair: %s <-> %s | num results: %zu",
      entry.first.first.c_str(),
      entry.first.second.c_str(),
      entry.second.size());

    if (
        (pair.first == link1 && pair.second == link2) ||
        (pair.first == link2 && pair.second == link1)
    )
    {
        for (const auto& d : vec)
        {
        RCLCPP_INFO(get_logger(),
            "Distance %s <-> %s: %f",
            pair.first.c_str(),
            pair.second.c_str(),
            d.distance);

        RCLCPP_INFO(get_logger(),
            "Point A: %f %f %f",
            d.nearest_points[0].x(),
            d.nearest_points[0].y(),
            d.nearest_points[0].z());

        RCLCPP_INFO(get_logger(),
            "Point B: %f %f %f",
            d.nearest_points[1].x(),
            d.nearest_points[1].y(),
            d.nearest_points[1].z());
        }
    }
    }

    return true; //!col_res.collision;
  }

  bool computeIKWithCollision(const geometry_msgs::msg::Pose& pose)
  {
    robot_state_->setToDefaultValues();

    return robot_state_->setFromIK(
      joint_model_group_,
      pose,
      0.0,
      [this](moveit::core::RobotState* state,
             const moveit::core::JointModelGroup*,
             const double*)
      {
        return this->isStateValid(state);
      }
    );
  }

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

    bool success = computeIKWithCollision(target_pose);

    if (true)
    {
      RCLCPP_INFO(get_logger(), "IK SUCCESS");

      std::vector<double> joint_values;
      robot_state_->copyJointGroupPositions(joint_model_group_, joint_values);

      // Get joint names from the group
      const std::vector<std::string>& joint_names = joint_model_group_->getVariableNames();

      for (size_t i = 0; i < joint_values.size(); ++i)
      {
        RCLCPP_INFO(get_logger(),
              "%s: %f",
              joint_names[i].c_str(),
              joint_values[i]);
      }
    }
    if (!success)
    {
      RCLCPP_WARN(get_logger(), "IK FAILED");
    }

    RCLCPP_INFO(get_logger(), "Removing obstacle...");
    removeBoxObstacle();
  }

  // Members
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

  // Run initialization AFTER node is fully constructed
  std::thread init_thread([node]() {
    node->initialize();
  });

  rclcpp::spin(node);

  init_thread.join();

  rclcpp::shutdown();
  return 0;
}