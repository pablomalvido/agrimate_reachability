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
#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <random>

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <custom_interfaces/srv/collision_cost.hpp>

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

    normalization_loaded_ = loadNormalization();
    col_cost_client_ = this->create_client<custom_interfaces::srv::CollisionCost>("collision_cost");
    while (!col_cost_client_->wait_for_service(std::chrono::seconds(1)))
    {
      RCLCPP_WARN(get_logger(), "Waiting for collision_cost service...");
    }

    // Update for each "vineyard" position if they move
    try
    {
      geometry_msgs::msg::TransformStamped tf_world_to_vineyard = 
        tf_buffer_.lookupTransform(
          "vineyard_base",   // target frame
          "world",      // source frame
          tf2::TimePointZero);

      T_vineyard_world = tf2::transformToEigen(tf_world_to_vineyard.transform);
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_WARN(get_logger(), "TF error: %s", ex.what());
      return ;
    }

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
    //std::vector<double> dist_to_limits;

    double joint_centering_cost;
    double col_distance;
    double col_distance_norm;
    double col_interference;
    double manipulability;
    double manipulability_norm;
    double manip_world_y;
    double manip_world_y_norm;
    double manip_tool_neg_z;
    double manip_tool_neg_z_norm;
    double manip_world_y_fast;
    double manip_tool_neg_z_fast;
  };

  struct MeanMetrics
  {
    double joint_centering_cost = 0.0;
    double col_distance_norm = 0.0;
    double col_interference = 0.0;
    double manipulability_norm = 0.0;
    double manip_world_y_norm = 0.0;
    double manip_tool_neg_z_norm = 0.0;
  };

  IKMetrics computeMetrics()
  {
    IKMetrics m;

    // =========================
    // Distance to collision cost
    // =========================

    //m.col_distance = col_distance_;

    if (col_distance_ >= threshold_col)
    {
      m.col_distance_norm = 0.0;
    }
    else
    {
      double d = col_distance_;
      m.col_distance_norm =
        std::exp(-(d / (threshold_col - d)));
    }

    // =========================
    // Collision interference
    // =========================

    m.col_interference = computeCollisionCost();

    // =========================
    // Joint values
    // =========================
    robot_state_->copyJointGroupPositions(
      joint_model_group_,
      m.joint_values);

    const auto& joints = joint_model_group_->getActiveJointModels();

    double H = 0.0;

    for (size_t i = 0; i < joints.size(); ++i)
    {
      const auto& b = joints[i]->getVariableBounds()[0];

      double q = m.joint_values[i];

      double q_min = b.min_position_;
      double q_max = b.max_position_;

      double q_mid = 0.5 * (q_min + q_max);

      // normalized joint position in [-1, 1]
      double normalized =
        2.0 * (q - q_mid) / (q_max - q_min);

      // squared contribution
      H += normalized * normalized;
    }

    // normalize final metric to [0,1]
    H /= static_cast<double>(joints.size());

    // store result
    m.joint_centering_cost = H;

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
    double w_scaled = std::sqrt(JJt.determinant()) / std::pow(robot_length_, 3);

    if (normalization_loaded_)
    {
      m.manipulability_norm = 1.0 - std::exp(-k_yosh_ * w_scaled);
    }
    else
    {
      m.manipulability = w_scaled;
    }

    // =========================
    // Directional manipulability
    // =========================

    // --- WORLD -Y
    Eigen::VectorXd d_world = Eigen::VectorXd::Zero(6);
    d_world(1) = 1.0; // +Y direction
    //m.manip_world_y_fast = (J.transpose() * d_world).norm();

    double manip_world_y =
      1.0 / std::sqrt((d_world.transpose() * JJt.inverse() * d_world)(0,0));

    double wwy_scaled = manip_world_y / robot_length_;

    if (normalization_loaded_)
    {
      m.manip_world_y_norm = 1.0 - std::exp(-k_world_y_ * wwy_scaled);
    }
    else
    {
      m.manip_world_y = wwy_scaled;
    }

    // --- TOOL -Z
    Eigen::Isometry3d T =
      robot_state_->getGlobalLinkTransform(
        joint_model_group_->getLinkModelNames().back());

    Eigen::Vector3d z_tool = T.rotation() * Eigen::Vector3d(0, 0, -1);

    Eigen::VectorXd d_tool = Eigen::VectorXd::Zero(6);
    d_tool.head<3>() = z_tool;

    //m.manip_tool_neg_z_fast = (J.transpose() * d_tool).norm();

    double manip_tool_neg_z =
      1.0 / std::sqrt((d_tool.transpose() * JJt.inverse() * d_tool)(0,0));

    double wtz_scaled = manip_tool_neg_z / robot_length_;

    if (normalization_loaded_)
    {
      m.manip_tool_neg_z_norm = 1.0 - std::exp(-k_world_y_ * wtz_scaled);
    }
    else
    {
      m.manip_tool_neg_z = wtz_scaled;
    }

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


  std::vector<geometry_msgs::msg::Point>
  interpolatePoints(
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    int samples)
  {
    std::vector<geometry_msgs::msg::Point> points;

    for (int i = 0; i <= samples; ++i)
    {
      double t = static_cast<double>(i) / samples;

      Eigen::Vector3d p = (1.0 - t) * p1 + t * p2;

      geometry_msgs::msg::Point msg;
      msg.x = p.x();
      msg.y = p.y();
      msg.z = p.z();

      points.push_back(msg);
    }

    return points;
  }

  double computeCollisionCost()
  {
    std::vector<geometry_msgs::msg::Point> sampled_points;
    for (const auto& pair : col_link_pairs)
    {
      Eigen::Isometry3d T1_world =
        robot_state_->getGlobalLinkTransform(pair.first);

      Eigen::Isometry3d T2_world =
        robot_state_->getGlobalLinkTransform(pair.second);

      Eigen::Isometry3d T1_vineyard =
          T_vineyard_world * T1_world;

      Eigen::Isometry3d T2_vineyard =
          T_vineyard_world * T2_world;

      Eigen::Vector3d p1 = T1_vineyard.translation();
      Eigen::Vector3d p2 = T2_vineyard.translation();

      auto pts = interpolatePoints(p1, p2, 10);

      sampled_points.insert(
        sampled_points.end(),
        pts.begin(),
        pts.end());
    }
    auto request = std::make_shared<custom_interfaces::srv::CollisionCost::Request>();

    request->points = sampled_points;

    auto future = col_cost_client_->async_send_request(request);

    auto status = future.wait_for(std::chrono::milliseconds(500));

    if (status != std::future_status::ready)
    {
      RCLCPP_WARN(get_logger(), "Collision service timeout");
      return 1.0;
    }

    auto response = future.get();

    return response->cost;
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
          "Manipulability: %f | World +Y: %f | Tool -Z: %f | Collision distance: %f | Centering cost: %f",
          metrics.manipulability,
          metrics.manip_world_y,
          metrics.manip_tool_neg_z,
          metrics.col_distance_norm,
          metrics.joint_centering_cost);

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

    if (!normalization_loaded_)
    {
      computeNormalizationConstants();
      saveNormalization();
      computeNormalizeMetrics();
    }

    //Logs summary
    // RCLCPP_INFO(get_logger(),
    //   "========== METRICS LIST ==========");
    // for (size_t i = 0; i < metrics_list_.size(); ++i)
    // {
    //   const auto& m = metrics_list_[i];

    //   std::ostringstream joints_ss;

    //   for (size_t j = 0; j < m.joint_values.size(); ++j)
    //   {
    //     joints_ss << m.joint_values[j];

    //     if (j < m.joint_values.size() - 1)
    //       joints_ss << ", ";
    //   }

    //   RCLCPP_INFO(get_logger(),
    //     "Sample %zu", i);

    //   RCLCPP_INFO(get_logger(),
    //     "  Joint values: [%s]",
    //     joints_ss.str().c_str());

    //   RCLCPP_INFO(get_logger(),
    //     "  Joint centering cost: %f",
    //     m.joint_centering_cost);

    //   RCLCPP_INFO(get_logger(),
    //     "  Collision distance: %f",
    //     m.col_distance_norm);

    //   RCLCPP_INFO(get_logger(),
    //     "  Collision interference: %f",
    //     m.col_interference);

    //   RCLCPP_INFO(get_logger(),
    //     "  Manipulability norm: %f",
    //     m.manipulability_norm);

    //   RCLCPP_INFO(get_logger(),
    //     "  Manip world +Y norm: %f",
    //     m.manip_world_y_norm);

    //   RCLCPP_INFO(get_logger(),
    //     "  Manip tool -Z norm: %f",
    //     m.manip_tool_neg_z_norm);

    //   RCLCPP_INFO(get_logger(),
    //     "-----------------------------------");
    // }

    MeanMetrics mean_metrics =
      computeMeanMetrics(metrics_list_);

    logMeanMetrics(mean_metrics);

    double total_cost =
      computeTotalCost(
        mean_metrics,
        ik_success_count,
        total,
        plan_success_count,
        plan_total_count);

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


  MeanMetrics computeMeanMetrics(
    const std::vector<IKMetrics>& metrics_list)
  {
    MeanMetrics means;

    if (metrics_list.empty())
    {
      RCLCPP_WARN(get_logger(), "Metrics list is empty");
      return means;
    }

    for (const auto& m : metrics_list)
    {
      means.joint_centering_cost += m.joint_centering_cost;
      means.col_distance_norm += m.col_distance_norm;
      means.col_interference += m.col_interference;
      means.manipulability_norm += m.manipulability_norm;
      means.manip_world_y_norm += m.manip_world_y_norm;
      means.manip_tool_neg_z_norm += m.manip_tool_neg_z_norm;
    }

    const double N = static_cast<double>(metrics_list.size());

    means.joint_centering_cost /= N;
    means.col_distance_norm /= N;
    means.col_interference /= N;
    means.manipulability_norm /= N;
    means.manip_world_y_norm /= N;
    means.manip_tool_neg_z_norm /= N;

    return means;
  }


  double computeTotalCost(
    const MeanMetrics& means,
    int ik_success_count,
    int total_ik,
    int plan_success_count,
    int total_plan)
  {
    // =========================
    // Normalize weights
    // =========================
    double weight_sum = 0.0;

    for (const auto& kv : metric_weights_)
    {
      weight_sum += kv.second;
    }

    if (weight_sum <= 1e-9)
    {
      RCLCPP_ERROR(get_logger(), "Invalid weight sum");
      return 0.0;
    }

    const double w_joint =
      metric_weights_.at("joint_centering_cost") / weight_sum;

    const double w_col_dist =
      metric_weights_.at("col_distance_norm") / weight_sum;

    const double w_col_interference =
      metric_weights_.at("col_interference") / weight_sum;

    const double w_manip =
      metric_weights_.at("manipulability_norm") / weight_sum;

    const double w_world_y =
      metric_weights_.at("manip_world_y_norm") / weight_sum;

    const double w_tool_z =
      metric_weights_.at("manip_tool_neg_z_norm") / weight_sum;

    // =========================
    // CHECK!!!! AND MAKE THIS CLEARER AT SOME POINT
    // Convert costs into quality metrics
    // Higher manipulability = lower cost
    // Farther from collisions = lower collision cost
    // =========================
    double joint_centering_quality = 1.0 - means.joint_centering_cost;
    double col_distance_quality = 1.0 - means.col_distance_norm;
    double col_interference_quality = 1.0 - means.col_interference;
    double manip_quality = means.manipulability_norm;
    double world_y_quality = means.manip_world_y_norm;
    double tool_z_quality = means.manip_tool_neg_z_norm;

    // =========================
    // Weighted metric score
    // =========================
    double metric_score =
        w_joint * joint_centering_quality +
        w_col_dist * col_distance_quality +
        w_col_interference * col_interference_quality +
        w_manip * manip_quality +
        w_world_y * world_y_quality +
        w_tool_z * tool_z_quality;

    // =========================
    // Success ratios
    // =========================
    double ik_ratio =
      (total_ik > 0)
        ? static_cast<double>(ik_success_count) / total_ik
        : 0.0;

    double planning_ratio =
      (total_plan > 0)
        ? static_cast<double>(plan_success_count) / total_plan
        : 0.0;

    // =========================
    // Final total cost
    // =========================
    double total_cost =
      metric_score * ik_ratio * planning_ratio;

    // =========================
    // Logs
    // =========================
    RCLCPP_INFO(get_logger(),
      "Metric score: %f",
      metric_score);

    RCLCPP_INFO(get_logger(),
      "IK success ratio: %f",
      ik_ratio);

    RCLCPP_INFO(get_logger(),
      "Planning success ratio: %f",
      planning_ratio);

    RCLCPP_INFO(get_logger(),
      "========== TOTAL COST (0 => Worst, 1 => Best) ==========");

    RCLCPP_INFO(get_logger(),
      "TOTAL COST: %f",
      total_cost);

    RCLCPP_INFO(get_logger(),
      "================================");

    return total_cost;
  }

  void logMeanMetrics(const MeanMetrics& means){

    RCLCPP_INFO(get_logger(),
      "========== MEAN METRICS ==========");

    RCLCPP_INFO(get_logger(),
      "Mean joint centering cost: %f",
      means.joint_centering_cost);

    RCLCPP_INFO(get_logger(),
      "Mean collision distance cost: %f",
      means.col_distance_norm);

    RCLCPP_INFO(get_logger(),
      "Mean collision interference: %f",
      means.col_interference);

    RCLCPP_INFO(get_logger(),
      "Mean manipulability norm: %f",
      means.manipulability_norm);

    RCLCPP_INFO(get_logger(),
      "Mean world +Y manipulability norm: %f",
      means.manip_world_y_norm);

    RCLCPP_INFO(get_logger(),
      "Mean tool -Z manipulability norm: %f",
      means.manip_tool_neg_z_norm);

    RCLCPP_INFO(get_logger(),
      "==================================");
  }

  void computeNormalizationConstants(){
    if (metrics_list_.empty())
    {
      RCLCPP_WARN(get_logger(), "No metrics available for normalization");
      return;
    }

    double sum_yosh = 0.0;
    double sum_world_y = 0.0;
    double sum_tool_z = 0.0;

    for (const auto& m : metrics_list_)
    {
      sum_yosh += m.manipulability;
      sum_world_y += m.manip_world_y;
      sum_tool_z += m.manip_tool_neg_z;
    }

    double mean_yosh =
      sum_yosh / static_cast<double>(metrics_list_.size());

    double mean_world_y =
      sum_world_y / static_cast<double>(metrics_list_.size());

    double mean_tool_z =
      sum_tool_z / static_cast<double>(metrics_list_.size());

    // k = 1 / mean
    k_yosh_ = 1.0 / mean_yosh;
    k_world_y_ = 1.0 / mean_world_y;
    k_tool_z_ = 1.0 / mean_tool_z;

    RCLCPP_INFO(get_logger(),
      "Computed normalization constants:");

    RCLCPP_INFO(get_logger(),
      "k_yosh    = %f", k_yosh_);

    RCLCPP_INFO(get_logger(),
      "k_world_y = %f", k_world_y_);

    RCLCPP_INFO(get_logger(),
      "k_tool_z  = %f", k_tool_z_);
  }

  void computeNormalizeMetrics(){
    for (auto& m : metrics_list_)
    {  
      m.manipulability_norm =
        1.0 - std::exp(
          -k_yosh_ * m.manipulability);

      m.manip_world_y_norm =
        1.0 - std::exp(
          -k_world_y_ * m.manip_world_y);

      m.manip_tool_neg_z_norm =
        1.0 - std::exp(
          -k_tool_z_ * m.manip_tool_neg_z);
    }

    RCLCPP_INFO(get_logger(),
      "Normalized %zu metric samples",
      metrics_list_.size());
  }

  void saveNormalization()
  {
    YAML::Emitter out;

    out << YAML::BeginMap;
    out << YAML::Key << "manipulability";
    out << YAML::BeginMap;

    out << YAML::Key << "k_yosh" << YAML::Value << k_yosh_;
    out << YAML::Key << "k_world_y" << YAML::Value << k_world_y_;
    out << YAML::Key << "k_tool_z" << YAML::Value << k_tool_z_;

    out << YAML::EndMap;
    out << YAML::EndMap;

    std::ofstream fout(yaml_src_path_);
    fout << out.c_str();

    RCLCPP_INFO(get_logger(), "Saved normalization constants");
  }

  bool loadNormalization()
  {
    if (!std::filesystem::exists(yaml_src_path_))
      return false;

    YAML::Node config = YAML::LoadFile(yaml_src_path_);

    k_yosh_ = config["manipulability"]["k_yosh"].as<double>();
    k_world_y_ = config["manipulability"]["k_world_y"].as<double>();
    k_tool_z_ = config["manipulability"]["k_tool_z"].as<double>();

    RCLCPP_INFO(get_logger(), "Loaded normalization constants");

    return true;
  }

  // Members
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  planning_scene::PlanningScenePtr planning_scene_;
  const moveit::core::JointModelGroup* joint_model_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string base_frame_;
  double col_distance_;
  std::vector<std::pair<std::string, std::string>> col_link_pairs =
  {
    {"forearm_link", "wrist_1_link"},
    {"wrist_1_link", "wrist_2_link"},
    {"wrist_2_link", "wrist_3_link"},
    {"wrist_3_link", "tool0"} //Change "tool0" to "tcp_link"
  };
  rclcpp::Client<custom_interfaces::srv::CollisionCost>::SharedPtr col_cost_client_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  Eigen::Isometry3d T_vineyard_world;

  double k_yosh_ = -1.0;
  double k_world_y_ = -1.0;
  double k_tool_z_ = -1.0;
  bool normalization_loaded_ = false;
  const double robot_length_ = 0.85;
  double threshold_col = 0.15;

  const std::unordered_map<std::string, double> metric_weights_ =
  {
    {"joint_centering_cost", 0.25},
    {"col_distance_norm",    0.15},
    {"col_interference",     0.20},
    {"manipulability_norm",  0.10},
    {"manip_world_y_norm",   0.15},
    {"manip_tool_neg_z_norm",0.15}
  };

  std::vector<IKMetrics> metrics_list_;
  std::string pkg_path = ament_index_cpp::get_package_share_directory("moveit_cpp_demo");
  std::string manip_metrics_path = pkg_path + "/config/manipulability.yaml";
  std::string yaml_src_path_ = "/home/rosdev/ros2_ws/src/moveit_cpp_demo/config/manipulability.yaml";
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