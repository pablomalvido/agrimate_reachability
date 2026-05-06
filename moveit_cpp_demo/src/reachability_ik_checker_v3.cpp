#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>
//#include <moveit/planning_interface/move_group_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <random>

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

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "ur_manipulator");

    move_group_->setPlanningTime(0.6);
    move_group_->setNumPlanningAttempts(50);


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
  struct IKMetrics
  {
    std::vector<double> joint_values;
    std::vector<double> dist_to_limits;

    double col_distance;
    double manipulability;
    double manip_world_y;
    double manip_tool_neg_z;
    double manip_world_y_fast;
    double manip_tool_neg_z_fast;
  };

  IKMetrics computeMetrics()
  {
    IKMetrics m;

    m.col_distance = col_distance_;

    // =========================
    // Joint values
    // =========================
    robot_state_->copyJointGroupPositions(joint_model_group_, m.joint_values);

    // =========================
    // Joint limits distance
    // =========================
    const auto& bounds = joint_model_group_->getActiveJointModels();

    for (size_t i = 0; i < bounds.size(); ++i)
    {
      const auto& b = bounds[i]->getVariableBounds()[0];

      double q = m.joint_values[i];
      double d = std::min(q - b.min_position_, b.max_position_ - q);

      m.dist_to_limits.push_back(d);
    }

    // =========================
    // Jacobian
    // =========================
    Eigen::MatrixXd J;

    robot_state_->getJacobian(
      joint_model_group_,
      robot_state_->getLinkModel(joint_model_group_->getLinkModelNames().back()),
      Eigen::Vector3d::Zero(),
      J
    );

    Eigen::MatrixXd JJt = J * J.transpose();

    // =========================
    // Yoshikawa
    // =========================
    m.manipulability = std::sqrt(JJt.determinant());

    // =========================
    // Directional manipulability
    // =========================

    // --- WORLD -Y
    Eigen::VectorXd d_world = Eigen::VectorXd::Zero(6);
    d_world(1) = 1.0; // +Y direction

    m.manip_world_y =
      1.0 / std::sqrt((d_world.transpose() * JJt.inverse() * d_world)(0,0));

    //m.manip_world_y_fast = (J.transpose() * d_world).norm();

    // --- TOOL -Z
    Eigen::Isometry3d T =
      robot_state_->getGlobalLinkTransform(
        joint_model_group_->getLinkModelNames().back());

    Eigen::Vector3d z_tool = T.rotation() * Eigen::Vector3d(0, 0, -1);

    Eigen::VectorXd d_tool = Eigen::VectorXd::Zero(6);
    d_tool.head<3>() = z_tool;

    m.manip_tool_neg_z =
      1.0 / std::sqrt((d_tool.transpose() * JJt.inverse() * d_tool)(0,0));

    //m.manip_tool_neg_z_fast = (J.transpose() * d_tool).norm();

    // Logs to verify correct execution
    /*
    RCLCPP_INFO(get_logger(),
      "Robot model frame (Jacobian frame): %s",
      robot_model_->getModelFrame().c_str());

    RCLCPP_INFO(get_logger(),
      "d_world Y: [%f, %f, %f, %f, %f, %f]",
      d_world(0), d_world(1), d_world(2),
      d_world(3), d_world(4), d_world(5));

    RCLCPP_INFO(get_logger(),
      "d_tool Z: [%f, %f, %f, %f, %f, %f]",
      d_tool(0), d_tool(1), d_tool(2),
      d_tool(3), d_tool(4), d_tool(5));
    */

    return m;
  }

  bool planToIKSolution()
  {
    // Get IK solution from robot_state_
    std::vector<double> joint_values;
    robot_state_->copyJointGroupPositions(joint_model_group_, joint_values);

    // 🔹 Set start state (predefined config)
    moveit::core::RobotState start_state(robot_model_);
    start_state.setToDefaultValues(joint_model_group_, "start_config");

    move_group_->setStartState(start_state);

    // 🔹 Set goal = IK solution
    move_group_->setJointValueTarget(joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    bool success = (move_group_->plan(plan) ==
                    moveit::core::MoveItErrorCode::SUCCESS);

    return success;
  }

  bool planToPose(const geometry_msgs::msg::Pose& target_pose)
  {
    // 🔹 Set start state (predefined config)
    moveit::core::RobotState start_state(robot_model_);
    start_state.setToDefaultValues(joint_model_group_, "start_config");

    move_group_->setStartState(start_state);

    // 🔹 Set goal = pose
    move_group_->setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    bool success = (move_group_->plan(plan) ==
                    moveit::core::MoveItErrorCode::SUCCESS);

    // Always good practice
    move_group_->clearPoseTargets();

    return success;
  }

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
      col_distance_ = res.distance;
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
        0.0   // solver timeout
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
    auto t_start = std::chrono::high_resolution_clock::now();
    // printCollisionMatrix();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.2); // this percentage of poses will be evaluated for motion planning

    int ik_success_count = 0;
    int plan_success_count = 0;
    int plan_total_count = 0;
    int total = 100;

    std::vector<IKMetrics> metrics_list_;

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
        ik_success_count++;
        RCLCPP_INFO(this->get_logger(), "IK success: %s", frame_name.c_str());

        IKMetrics metrics = computeMetrics();
        metrics_list_.push_back(metrics);

        /*RCLCPP_INFO(this->get_logger(),
          "Manipulability: %f | World -Y: %f | World -Y Fast: %f | Tool -Z: %f | Tool -Z Fast: %f",
          metrics.manipulability,
          metrics.manip_world_y,
          metrics.manip_world_y_fast,
          metrics.manip_tool_neg_z,
          metrics.manip_tool_neg_z_fast);*/

        RCLCPP_INFO(this->get_logger(),
          "Manipulability: %f | World +Y: %f | Tool -Z: %f | Collision distance: %f",
          metrics.manipulability,
          metrics.manip_world_y,
          metrics.manip_tool_neg_z,
          metrics.col_distance);

        if (dist(gen))
        {
          plan_total_count++;
          // PLAN ONLY IF IK SUCCEEDED
          bool plan_success = false;

          //if (planToIKSolution())
          if (planToPose(pose))
          {
            plan_success = true;
          }

          if (plan_success)
          {
            plan_success_count++;
            RCLCPP_INFO(this->get_logger(), "Planning SUCCESS: %s", frame_name.c_str());
          }
          else
          {
            RCLCPP_WARN(this->get_logger(), "Planning FAILED: %s", frame_name.c_str());
          }
        }
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "IK failed: %s", frame_name.c_str());
      }
      removeBoxObstacle();
    }

    RCLCPP_INFO(this->get_logger(),
      "IK success: %d / %d | Planning success: %d / %d",
      ik_success_count, total,
      plan_success_count, plan_total_count);

    //RCLCPP_INFO(this->get_logger(), "Stored %zu IK metric samples", metrics_list_.size());

    auto t_end = std::chrono::high_resolution_clock::now();

    double total_time =
      std::chrono::duration<double>(t_end - t_start).count();

    RCLCPP_INFO(this->get_logger(),
      "Total computation time: %.6f seconds", total_time);
  }

  // Members
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  planning_scene::PlanningScenePtr planning_scene_;
  const moveit::core::JointModelGroup* joint_model_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string base_frame_;
  double col_distance_;

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