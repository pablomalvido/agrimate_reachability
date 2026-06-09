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
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit/robot_state/conversions.h>

#include "simplified_bayesian_optimizer.hpp"

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

    robot_state_pub_ = this->create_publisher<moveit_msgs::msg::DisplayRobotState>("/display_robot_state", 10);
    publisher_passive_joints_ = this->create_publisher<sensor_msgs::msg::JointState>("/passive_joint_commands", 10);

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

    RCLCPP_INFO(
      this->get_logger(),
      "End effector link: %s",
      move_group_->getEndEffectorLink().c_str());

    //LOAD FROM YAML
    double y_min = -1.0; //m
    double y_max = -0.5; //m
    double rot_min = -0.8; //rad
    double rot_max = 1.05; //rad
    double z_min = -0.05; //m
    double z_max = 0.35; //m
    SimplifiedBayesianOptimizer optimizer(
      y_min, y_max,     // Y limits
      z_min, z_max,      // Z limits
      rot_min, rot_max);    // roll limits

    std::vector<PlacementSample> history;

    // std::vector<double> config = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // setRobotConfiguration(config);

    // std::vector<double> current;

    // robot_state_->copyJointGroupPositions(
    //     joint_model_group_,
    //     current);

    // RCLCPP_INFO(
    //     get_logger(),
    //     "Joints = %f, %f, %f, %f, %f, %f",
    //     current[0], current[1], current[2], current[3], current[4], current[5]);
    
    for (int iter = 0; iter < 100; ++iter)
    {
        Placement p = optimizer.proposeNext(history);

        new_config_ = computeRobotConfig(p.z, p.roll, z_min, z_max); //Computes the joint values
        applyRobotPlacement(p.y, p.z, p.roll); //Updates robot base position

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

        bool valid_state = setRobotConfiguration(new_config_); //Updates the joint values and robot state (not visually though)
        if (!valid_state){
          RCLCPP_WARN(get_logger(), "State not valid");

          // history.push_back({
          //     p.y,
          //     p.z,
          //     p.roll,
          //     0.0,
          //     iter,
          // });

          RCLCPP_INFO(
            get_logger(),
            "### CONIFG: Vineyard: %f, Z: %f, Rot: %f, Shoulder: %f",
            p.y, p.z, p.roll, new_config_[1]);

          continue;
        }

        // const auto& tf =
        //     robot_state_->getGlobalLinkTransform("base_link");

        // RCLCPP_INFO(get_logger(),
        //             "Base position: %.3f %.3f %.3f",
        //             tf.translation().x(),
        //             tf.translation().y(),
        //             tf.translation().z());

        // for (const auto& name : robot_state_->getVariableNames())
        // {
        //     RCLCPP_INFO(get_logger(), "%s", name.c_str());
        // }

        ReachabilityConfig config_prune;
        config_prune.pose_prefix = "pruning_pose_";
        config_prune.enable_planning = true;
        config_prune.planning_probability = 0.3;
        config_prune.obstacle = "relative";
        double score_prune = evaluateReachability(config_prune).total_cost;

        //Evaluate also for scan and for grasp. Then the score will be the weighted average of these. CHANGE the part of the collision obstacle, simply spawn a collision obstacle at the vineyard base of certain dimensions
        ReachabilityConfig config_scan;
        config_scan.pose_prefix = "scaning_pose_";
        config_scan.enable_planning = true;
        config_scan.planning_probability = 0.3;
        config_scan.obstacle = "fixed";
        double score_scan = evaluateReachability(config_scan).total_cost;

        ReachabilityConfig config_grasp;
        config_grasp.pose_prefix = "grasping_pose_";
        config_grasp.enable_planning = true;
        config_grasp.planning_probability = 0.3;
        config_grasp.obstacle = "fixed";
        double score_grasp = evaluateReachability(config_grasp).total_cost;

        double score = 0.5*score_prune + 0.3*score_scan + 0.2*score_grasp;

        history.push_back({
            p.y,
            p.z,
            p.roll,
            score_prune,
            score_scan,
            score_grasp,
            score,
            iter,
        });

        std::cout
            << "Iter " << iter
            << " score=" << score
            << std::endl;
    }

    // Sort history by best score (higher = better)
    std::sort(history.begin(), history.end(),
              [](const auto &a, const auto &b) {
                  return a.score > b.score;
              });

    // Print top 
    std::cout << "\n===== TOP CONFIGURATIONS =====\n";

    for (size_t i = 0; i < std::min<size_t>(100, history.size()); ++i)
    {
        const auto &h = history[i];

        std::cout << "Rank " << i + 1
                  << " | Iteration=" << h.iter
                  << " | y=" << h.y
                  << " z=" << h.z
                  << " roll=" << h.roll
                  << " score=" << h.score
                  << " score_prune=" << h.score1
                  << " score_scan=" << h.score2
                  << " score_grasp=" << h.score3
                  << std::endl;
    }

    // Design space exploration: separate scripts that run different exploration policies (CEM, Gen Alg, etc.) to test them. Test all these with just one task and with no planning, to be faster.
    // CONFIG for each task: TF prefix, add collision obstacle and at which offset from the pose?, weight, planning % 
    // ReachabilityConfig config;
    // config.pose_prefix = "sample_pose_";
    // config.enable_planning = false;
    // config.planning_probability = 0.2;

    // auto result =
    //   evaluateReachability(config);

    // ReachabilityConfig config2;
    // config2.pose_prefix = "sample_pose_";
    // config2.enable_planning = false;
    // config2.planning_probability = 0.2;

    // auto result2 =
    //   evaluateReachability(config2);

    // RCLCPP_INFO(
    //   get_logger(),
    //   "FINAL SCORE: %f",
    //   result.total_cost);
  }


  int getHighestPoseIndex(const std::string& prefix)
  {
      std::vector<std::string> frames;
      tf_buffer_._getFrameStrings(frames);

      int max_index = -1;

      for (const auto& frame : frames)
      {
          if (frame.rfind(prefix, 0) == 0)  // starts with prefix
          {
              std::string suffix = frame.substr(prefix.size());

              try
              {
                  int idx = std::stoi(suffix);
                  max_index = std::max(max_index, idx);
              }
              catch (...)
              {
                  // Ignore names that are not prefix + integer
              }
          }
      }

      return max_index;
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

  struct ReachabilityConfig
  {
    std::string pose_prefix = "sample_pose_";
    bool enable_planning = true;
    double planning_probability = 0.2;
    std::string obstacle = "";
  };

  struct ReachabilityResult
  {
    int ik_success_count = 0;

    int planning_success_count = 0;

    int planning_total_count = 0;

    MeanMetrics mean_metrics;

    double total_cost = 0.0;

    std::vector<IKMetrics> metrics_list;
  };

  std::vector<double> computeRobotConfig(
    double slider_z,
    double cylinder_rot,
    double z_min,
    double z_max)
  {
    std::vector<double> q(6, 0.0);

    //--------------------------------------------------
    // Example mapping
    // Replace with your own equations
    //--------------------------------------------------

    q[0] = 0.0;
    //q[1] is done later
    q[2] = 1.93;
    q[3] = -2.96;
    q[4] = -1.75;
    q[5] = -1.6;

    double alpha = 0.0;

    if (slider_z > 0.0){
      alpha = (slider_z) / (z_max);
    }

    alpha = std::clamp(alpha, 0.0, 1.0);

    q[1] =  -1.35 + alpha * 1.35;
    q[1] += -cylinder_rot * 1.05;

    return q;
  }

  void applyRobotPlacement(double p_y, double p_z, double p_roll)
  {
    auto msg = sensor_msgs::msg::JointState();
    msg.name = {
        "world_to_vineyard",
        "platform_to_slider",
        "platform_to_cylinder"
    };

    msg.position = {
        p_y,
        p_z,
        p_roll
    };

    publisher_passive_joints_->publish(msg);

    robot_state_->setVariablePosition("platform_to_slider", p_z);
    robot_state_->setVariablePosition("platform_to_cylinder", p_roll);
    robot_state_->setVariablePosition("world_to_vineyard", p_y);

    robot_state_->update();
  }

  bool setRobotConfiguration(const std::vector<double>& joint_values)
  {
      if (joint_values.size() !=
          joint_model_group_->getVariableCount())
      {
          RCLCPP_ERROR(
            get_logger(),
            "Expected %u joints, got %zu",
            joint_model_group_->getVariableCount(),
            joint_values.size());

          return false;
      }

      //--------------------------------------------------
      // Update internal RobotState
      //--------------------------------------------------

      robot_state_->setJointGroupPositions(
          joint_model_group_,
          joint_values);

      robot_state_->update();

      std::vector<double> current;

      robot_state_->copyJointGroupPositions(
          joint_model_group_,
          current);

      // for(double q : current)
      // {
      //     RCLCPP_INFO(get_logger(), "%f", q);
      // }

      //--------------------------------------------------
      // Update Planning Scene state
      //--------------------------------------------------

      planning_scene_->setCurrentState(*robot_state_);

      //--------------------------------------------------
      // Update MoveGroup start state
      //--------------------------------------------------

      move_group_->setStartState(*robot_state_);

      //--------------------------------------------------
      // Publish state to RViz
      //--------------------------------------------------

      moveit_msgs::msg::DisplayRobotState msg;

      moveit::core::robotStateToRobotStateMsg(
          *robot_state_,
          msg.state);

      robot_state_pub_->publish(msg);

      // RCLCPP_INFO(get_logger(), "Robot configuration updated");

      return isStateValid(robot_state_.get());
  }

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
      m.manip_tool_neg_z_norm = 1.0 - std::exp(-k_tool_z_ * wtz_scaled);
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

    // // Set start state (predefined config)
    // moveit::core::RobotState start_state(robot_model_);
    // start_state.setToDefaultValues(joint_model_group_, "start_config");
    // move_group_->setStartState(start_state);
    setRobotConfiguration(new_config_);


    // Set goal = IK solution
    move_group_->setJointValueTarget(joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    bool success = (move_group_->plan(plan) ==
                    moveit::core::MoveItErrorCode::SUCCESS);

    return success;
  }

  bool planToPose(const geometry_msgs::msg::Pose& target_pose)
  {
    // // 🔹 Set start state (predefined config)
    // moveit::core::RobotState start_state(robot_model_);
    // start_state.setToDefaultValues(joint_model_group_, "start_config");
    // move_group_->setStartState(start_state);

    setRobotConfiguration(new_config_);

    // Set goal = pose
    move_group_->setPoseTarget(target_pose, "ee_link");

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
  void addFixedBoxObstacle()
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "temp_box";
    obj.header.frame_id = "vineyard_base";
    obj.pose.position.x = 0.0;
    obj.pose.position.y = 0.0;
    float dim_z = 0.7;
    obj.pose.position.z = dim_z/2.0;
    obj.pose.orientation.w = 1.0;

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

  void removeFixedBoxObstacle()
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "temp_box";
    obj.header.frame_id = "vineyard_base";
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
      // RCLCPP_WARN(get_logger(), "State in collision");
      // for (auto &contact : res.contacts) 
      // { 
      //     RCLCPP_WARN(get_logger(), "Collision: %s <-> %s", contact.first.first.c_str(), contact.first.second.c_str()); 
      // } 
    } else {
      // RCLCPP_INFO(get_logger(), "Distance to collision: %f", res.distance);
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
      // RCLCPP_INFO(get_logger(), "IK attempt %d", attempt + 1);

      // Different seed each time
      if (attempt==0)
      {
        //robot_state_->setToDefaultValues();
        setRobotConfiguration(new_config_);
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
        // RCLCPP_WARN(get_logger(), "IK solver failed");
        continue;
      }

      // Check collision AFTER IK
      if (isStateValid(robot_state_.get()))
      {
        // RCLCPP_INFO(get_logger(), "Valid solution found!");
        return true;
      }
      else
      {
        // RCLCPP_WARN(get_logger(), "Solution in collision, retrying...");
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


  ReachabilityResult evaluateReachability(const ReachabilityConfig& config)
  {
    auto t_start = std::chrono::high_resolution_clock::now();

    ReachabilityResult result;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::bernoulli_distribution plan_dist(
      config.planning_probability);

    int total_poses = getHighestPoseIndex(config.pose_prefix)+1;

    RCLCPP_INFO(
      get_logger(),
      "Checking %d poses with prefix '%s'",
      total_poses,
      config.pose_prefix.c_str());

    // IMPORTANT:
    // clear previous run metrics
    metrics_list_.clear();

    for (int i = 0; i < total_poses; ++i)
    {
      std::string frame_name =
        config.pose_prefix + std::to_string(i);

      geometry_msgs::msg::TransformStamped transform;

      try
      {
        transform = tf_buffer_.lookupTransform(
          base_frame_,
          frame_name,
          tf2::TimePointZero);
      }
      catch (tf2::TransformException &ex)
      {
        RCLCPP_WARN(
          get_logger(),
          "TF not found for %s: %s",
          frame_name.c_str(),
          ex.what());

        continue;
      }

      geometry_msgs::msg::Pose pose;

      pose.position.x =
        transform.transform.translation.x;

      pose.position.y =
        transform.transform.translation.y;

      pose.position.z =
        transform.transform.translation.z;

      pose.orientation =
        transform.transform.rotation;

      if (config.obstacle=="relative"){
        addBoxObstacle(pose);
      }
      else if (config.obstacle=="fixed"){
        addFixedBoxObstacle();
      }

      bool found_ik =
        computeIKWithRetries(pose);

      if (found_ik)
      {
        result.ik_success_count++;

        IKMetrics metrics =
          computeMetrics();

        metrics_list_.push_back(metrics);

        if (config.enable_planning &&
            plan_dist(gen))
        {
          result.planning_total_count++;

          bool success =
            planToPose(pose);

          if (success)
          {
            result.planning_success_count++;
          }
        }
      }
      if (config.obstacle=="relative"){
        removeBoxObstacle();
      }
      else if (config.obstacle=="fixed"){
        removeFixedBoxObstacle();
      }
    }

    // =========================
    // NORMALIZATION
    // =========================

    if (!normalization_loaded_)
    {
      computeNormalizationConstants();

      saveNormalization();

      computeNormalizeMetrics();
    }

    // =========================
    // FINAL METRICS
    // =========================

    result.metrics_list = metrics_list_;

    result.mean_metrics =
      computeMeanMetrics(metrics_list_);

    result.total_cost =
      computeTotalCost(
        result.mean_metrics,
        result.ik_success_count,
        total_poses,
        result.planning_success_count,
        result.planning_total_count);

    logMeanMetrics(result.mean_metrics);

    auto t_end =
      std::chrono::high_resolution_clock::now();

    double total_time =
      std::chrono::duration<double>(
        t_end - t_start).count();

    RCLCPP_INFO(
      get_logger(),
      "Evaluation completed in %.3f sec",
      total_time);

    return result;
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
        : 1.0;

    double planning_ratio =
      (total_plan > 0)
        ? static_cast<double>(plan_success_count) / total_plan
        : 1.0;

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

    return total_cost;
  }

  void logMeanMetrics(const MeanMetrics& means){

    RCLCPP_INFO(get_logger(),
      "========== MEAN METRICS ==========");

    RCLCPP_INFO(get_logger(),
      "Mean joint centering quality: %f",
      (1 - means.joint_centering_cost));

    RCLCPP_INFO(get_logger(),
      "Mean collision distance quality: %f",
      (1 - means.col_distance_norm));

    RCLCPP_INFO(get_logger(),
      "Mean collision quality: %f",
      (1 - means.col_interference));

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

  rclcpp::Publisher<moveit_msgs::msg::DisplayRobotState>::SharedPtr robot_state_pub_;
  std::vector<double> new_config_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_passive_joints_;

  std::string base_frame_;
  double col_distance_;
  std::vector<std::pair<std::string, std::string>> col_link_pairs =
  {
    {"forearm_link", "wrist_1_link"},
    {"wrist_1_link", "wrist_2_link"},
    {"wrist_2_link", "wrist_3_link"},
    {"wrist_3_link", "tool0"},
    {"tool0", "ee_link"} //Change "tool0" to "tcp_link"
  };
  rclcpp::Client<custom_interfaces::srv::CollisionCost>::SharedPtr col_cost_client_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  Eigen::Isometry3d T_vineyard_world;

  double k_yosh_ = -1.0; //Should this be global (member)? Maybe it could be but with a different k for every task (this could be a struct with a k value for each task)?
  double k_world_y_ = -1.0;
  double k_tool_z_ = -1.0;
  bool normalization_loaded_ = false;
  const double robot_length_ = 0.85;
  double threshold_col = 0.15;

  // Do not need to add to 1, it is later normalized
  const std::unordered_map<std::string, double> metric_weights_ =
  {
    {"joint_centering_cost", 0.3},
    {"col_distance_norm",    0.1},
    {"col_interference",     0.1},
    {"manipulability_norm",  0.2},
    {"manip_world_y_norm",   0.2},
    {"manip_tool_neg_z_norm",0.3}
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