#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/conversions.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <custom_interfaces/srv/collision_cost.hpp>

#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>


// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Placement
{
    // In your current application:
    //
    // values[0] = Y
    // values[1] = Z
    // values[2] = roll
    //
    std::vector<double> values;

    // Grid indices
    int iy = 0;
    int iz = 0;
    int iroll = 0;
};


struct ConfigurationCandidate
{
    std::vector<double> joints;

    // Planning
    double planning_success_ratio = 0.0;

    // Other metrics
    double joint_centering_quality = 0.0;
    double collision_distance_quality = 0.0;
    double collision_quality = 0.0;
    double manipulability = 0.0;
    double manipulability_world_y = 0.0;
    double manipulability_tool_neg_z = 0.0;

    // Overall quality BEFORE spatial smoothing.
    double quality_score =
        -std::numeric_limits<double>::infinity();

    bool valid = false;
};


struct PlacementConfig
{
    Placement placement;

    // Several good configurations are preserved.
    //
    // candidates[0] = locally best
    // candidates[1] = second best
    // ...
    std::vector<ConfigurationCandidate> candidates;

    // Candidate selected after smoothing.
    int selected_candidate = -1;
};


// ============================================================================
// NODE
// ============================================================================

class IKReachabilityNode : public rclcpp::Node
{
public:

    IKReachabilityNode()
    : Node("initial_configuration_generator"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_),
      rng_(std::random_device{}())
    {
        RCLCPP_INFO(
            get_logger(),
            "Initial configuration generator created");
    }


    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    void initialize()
    {
        RCLCPP_INFO(
            get_logger(),
            "Initializing...");

        rclcpp::sleep_for(
            std::chrono::seconds(2));


        // ---------------------------------------------------------------------
        // Robot model
        // ---------------------------------------------------------------------

        robot_model_loader::RobotModelLoader model_loader(
            shared_from_this(),
            "robot_description");

        robot_model_ =
            model_loader.getModel();

        if (!robot_model_)
        {
            RCLCPP_ERROR(
                get_logger(),
                "Failed to load robot model");

            return;
        }


        robot_state_ =
            std::make_shared<moveit::core::RobotState>(
                robot_model_);

        robot_state_->setToDefaultValues();


        // ---------------------------------------------------------------------
        // Planning scene
        // ---------------------------------------------------------------------

        planning_scene_ =
            std::make_shared<planning_scene::PlanningScene>(
                robot_model_);


        // ---------------------------------------------------------------------
        // MoveIt group
        // ---------------------------------------------------------------------

        joint_model_group_ =
            robot_model_->getJointModelGroup(
                "ur_manipulator");

        if (!joint_model_group_)
        {
            RCLCPP_ERROR(
                get_logger(),
                "Joint model group not found");

            return;
        }


        base_frame_ =
            robot_model_->getModelFrame();


        // ---------------------------------------------------------------------
        // Move group
        // ---------------------------------------------------------------------

        move_group_ =
            std::make_shared<
                moveit::planning_interface::MoveGroupInterface>(
                    shared_from_this(),
                    "ur_manipulator");


        move_group_->setPlanningTime(0.1); //0.6
        move_group_->setNumPlanningAttempts(50);



        // ---------------------------------------------------------------------
        // Publishers
        // ---------------------------------------------------------------------

        robot_state_pub_ =
            create_publisher<
                moveit_msgs::msg::DisplayRobotState>(
                    "/display_robot_state",
                    10);


        publisher_passive_joints_ =
            create_publisher<
                sensor_msgs::msg::JointState>(
                    "/passive_joint_commands",
                    10);


        // ---------------------------------------------------------------------
        // Collision cost service
        // ---------------------------------------------------------------------

        col_cost_client_ =
            create_client<
                custom_interfaces::srv::CollisionCost>(
                    "collision_cost");


        while (!col_cost_client_->wait_for_service(
            std::chrono::seconds(1)))
        {
            RCLCPP_WARN(
                get_logger(),
                "Waiting for collision_cost service...");
        }


        // ---------------------------------------------------------------------
        // Robot length
        // ---------------------------------------------------------------------

        if (robot_type_ == "ur5" ||
            robot_type_ == "ur5e")
        {
            robot_length_ = 0.85;
        }
        else if (robot_type_ == "ur3" ||
                 robot_type_ == "ur3e")
        {
            robot_length_ = 0.5;
        }


        // ---------------------------------------------------------------------
        // Normalization constants
        // ---------------------------------------------------------------------

        normalization_loaded_ =
            loadNormalization();

        // ---------------------------------------------------------------------
        // Generate offline table
        // ---------------------------------------------------------------------

        generateOfflineConfigurationTable();

        removeFixedBoxObstacle();
        RCLCPP_INFO(
            get_logger(),
            "Removed obstacle");

        RCLCPP_INFO(
            get_logger(),
            "Offline configuration table generation finished");
    }


private:

    // =========================================================================
    // OFFLINE TABLE GENERATION
    // =========================================================================

    void generateOfflineConfigurationTable()
    {
        RCLCPP_INFO(
            get_logger(),
            "==============================================");

        RCLCPP_INFO(
            get_logger(),
            "GENERATING OFFLINE CONFIGURATION TABLE");

        RCLCPP_INFO(
            get_logger(),
            "==============================================");


        // ---------------------------------------------------------------------
        // Placement grid
        //
        // These are your current dimensions:
        //
        // Y
        // Z
        // ROLL
        // ---------------------------------------------------------------------

        const double y_min = -1.0;
        const double y_max = -0.5;

        const double z_min = -0.05;
        const double z_max = 0.35;

        const double roll_min = -0.8;
        const double roll_max = 1.05;


        // Number of samples in each dimension.
        //
        // CHANGE THESE.
        //
        const int n_y = 1;
        const int n_z = 10;
        const int n_roll = 10;


        // ---------------------------------------------------------------------
        // Number of random configurations per placement.
        // ---------------------------------------------------------------------

        const int n_random_candidates = 200;


        // ---------------------------------------------------------------------
        // Number of candidates preserved per placement.
        // ---------------------------------------------------------------------

        const int K = 10;


        // ---------------------------------------------------------------------
        // Smoothing
        // ---------------------------------------------------------------------

        const int smoothing_iterations = 10;

        const double smoothness_weight = 1.0;


        // ---------------------------------------------------------------------
        // Generate grid
        // ---------------------------------------------------------------------

        bool first_ite = true;

        std::vector<Placement> grid;


        for (int iy = 0;
             iy < n_y;
             ++iy)
        {
            // double y =
            //     interpolateGrid(
            //         y_min,
            //         y_max,
            //         iy,
            //         n_y);

            double y = -0.75; // Fixed Y for now


            for (int iz = 0;
                 iz < n_z;
                 ++iz)
            {
                double z =
                    interpolateGrid(
                        z_min,
                        z_max,
                        iz,
                        n_z);


                for (int ir = 0;
                     ir < n_roll;
                     ++ir)
                {
                    double roll =
                        interpolateGrid(
                            roll_min,
                            roll_max,
                            ir,
                            n_roll);


                    Placement p;

                    p.values = {
                        y,
                        z,
                        roll
                    };

                    p.iy = iy;
                    p.iz = iz;
                    p.iroll = ir;


                    grid.push_back(
                        p);
                }
            }
        }


        RCLCPP_INFO(
            get_logger(),
            "Grid contains %zu placements",
            grid.size());


        // ---------------------------------------------------------------------
        // Generate candidates
        // ---------------------------------------------------------------------

        std::vector<PlacementConfig> table;

        table.reserve(
            grid.size());


        for (size_t i = 0;
             i < grid.size();
             ++i)
        {
            const Placement& placement =
                grid[i];


            RCLCPP_INFO(
                get_logger(),
                "########### Placement %zu / %zu: "
                "Y=%.3f Z=%.3f Roll=%.3f",
                i + 1,
                grid.size(),
                placement.values[0],
                placement.values[1],
                placement.values[2]);


            // -------------------------------------------------------------
            // Update robot placement
            // -------------------------------------------------------------

            applyRobotPlacement(
                placement.values[0],
                placement.values[1],
                placement.values[2]);

            if (first_ite){
                addFixedBoxObstacle();
                first_ite = false;
            }

            // -------------------------------------------------------------
            // Generate random candidates
            // -------------------------------------------------------------

            std::vector<ConfigurationCandidate> candidates;

            int valid_candidates = 0;
            for (int c = 0;
                 c < n_random_candidates;
                 ++c)
            {

                double percentage =
                    (100.0 * (i) / grid.size())+ (1.0 / grid.size() * (100.0 * (c) / n_random_candidates));

                RCLCPP_INFO(
                get_logger(),
                "########### Progress: %f ##########",
                percentage);

                RCLCPP_INFO(
                get_logger(),
                "Placement %zu / %zu, Candidate %d / %d",
                i + 1,
                grid.size(),
                c + 1,
                n_random_candidates);

                ConfigurationCandidate candidate;

                candidate.joints.resize(
                    joint_model_group_->getVariableCount()
                );

                if (!generateRandomConfiguration(
                        candidate.joints))
                {
                    continue;
                }


                // ---------------------------------------------------------
                // Check collision
                // ---------------------------------------------------------

                if (!isConfigurationValid(
                        candidate.joints))
                {
                    continue;
                }


                // ---------------------------------------------------------
                // End-effector workspace restriction
                // ---------------------------------------------------------

                if (!endEffectorInAllowedRegion(
                        candidate.joints))
                {
                    continue;
                }


                // ---------------------------------------------------------
                // Evaluate candidate
                // ---------------------------------------------------------

                evaluateCandidate(
                    candidate);


                if (candidate.valid)
                {
                    candidates.push_back(
                        std::move(candidate));
                    valid_candidates++;
                    RCLCPP_INFO(
                    get_logger(),
                    "########### Valid candidate found! Total valid: %d/%d, Planning success ratio: %f", valid_candidates, c+1, candidate.planning_success_ratio);
                }
            }


            // -------------------------------------------------------------
            // Sort candidates according to individual quality
            // -------------------------------------------------------------

            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const ConfigurationCandidate& a,
                   const ConfigurationCandidate& b)
                {
                    return a.quality_score >
                           b.quality_score;
                });


            // -------------------------------------------------------------
            // Keep top K
            // -------------------------------------------------------------

            if (candidates.size() >
                static_cast<size_t>(K))
            {
                candidates.resize(K);
            }


            PlacementConfig result;

            result.placement =
                placement;

            result.candidates =
                std::move(candidates);

            result.selected_candidate =
                -1;


            table.push_back(
                std::move(result));


            if (table.back().candidates.empty())
            {
                RCLCPP_WARN(
                    get_logger(),
                    "No valid candidates for this placement!");
            }
            else
            {
                RCLCPP_INFO(
                    get_logger(),
                    "  Valid candidates: %zu",
                    table.back().candidates.size());

                RCLCPP_INFO(
                    get_logger(),
                    "  Best local score: %.4f",
                    table.back().candidates[0].quality_score);
            }
        }

        // ============================================================
        // SAVE RAW TABLE
        // ============================================================

        saveRawConfigurationTable(table);

        // ---------------------------------------------------------------------
        // Smooth the table
        // ---------------------------------------------------------------------

        RCLCPP_INFO(
            get_logger(),
            "Starting spatial smoothing...");


        smoothConfigurationTable(
            table,
            smoothing_iterations,
            smoothness_weight);


        // ---------------------------------------------------------------------
        // Print selected configurations
        // ---------------------------------------------------------------------

        printFinalTable(
            table);


        // ---------------------------------------------------------------------
        // Save
        // ---------------------------------------------------------------------

        saveConfigurationTable(
            table);
    }


    // =========================================================================
    // GRID INTERPOLATION
    // =========================================================================

    double interpolateGrid(
        double min,
        double max,
        int index,
        int count)
    {
        if (count <= 1)
            return min;

        double alpha =
            static_cast<double>(index) /
            static_cast<double>(count - 1);

        return min +
               alpha * (max - min);
    }


    // =========================================================================
    // RANDOM CONFIGURATION
    // =========================================================================

    bool generateRandomConfiguration(
        std::vector<double>& joints)
    {
        // robot_state_->setToRandomPositions(
        //     joint_model_group_);


        // robot_state_->copyJointGroupPositions(
        //     joint_model_group_,
        //     joints);

        // ============================================================
        // Custom random-sampling bounds
        // ============================================================

        std::vector<double> random_lower = {
            0.0,      // joint 0 shoulder pan
            -1.57,       // joint 1 shoulder lift
            0.0,       // joint 2 elbow
            -1.75,      // joint 3  wrist 1
            0.0,       // joint 4  wrist 2
            0.0      // joint 5  wrist 3
        };

        std::vector<double> random_upper = {
            0.0,      // joint 0
            1.0,       // joint 1
            2.5,       // joint 2
            1.75,      // joint 3
            0.0,       // joint 4
            0.0      // joint 5
        };

        // ============================================================
        // Generate random configuration inside those bounds
        // ============================================================

        for (std::size_t i = 0; i < random_lower.size(); ++i)
        {
            std::uniform_real_distribution<double> distribution(
                random_lower[i],
                random_upper[i]
            );

            joints[i] = distribution(rng_);
        }

        RCLCPP_INFO(
            get_logger(),
            "Generated random configuration: "
            "[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
            joints[0], joints[1], joints[2],
            joints[3], joints[4], joints[5]);
        // ============================================================
        // Set the generated configuration in MoveIt
        // ============================================================

        robot_state_->setJointGroupPositions(
            joint_model_group_,
            joints
        );


        return
            joints.size() ==
            joint_model_group_->getVariableCount();
    }


    // =========================================================================
    // APPLY ROBOT PLACEMENT
    //
    // This is your existing implementation.
    // =========================================================================

    void applyRobotPlacement(
        double p_y,
        double p_z,
        double p_roll)
    {
        auto msg =
            sensor_msgs::msg::JointState();


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


        publisher_passive_joints_->publish(
            msg);


        robot_state_->setVariablePosition(
            "platform_to_slider",
            p_z);


        robot_state_->setVariablePosition(
            "platform_to_cylinder",
            p_roll);


        robot_state_->setVariablePosition(
            "world_to_vineyard",
            p_y);


        robot_state_->update();
    }


    // =========================================================================
    // SET ROBOT CONFIGURATION
    //
    // Adapted from your existing function.
    // =========================================================================

    bool setRobotConfiguration(
        const std::vector<double>& joint_values)
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


        robot_state_->setJointGroupPositions(
            joint_model_group_,
            joint_values);


        robot_state_->update();


        planning_scene_->setCurrentState(
            *robot_state_);


        move_group_->setStartState(
            *robot_state_);


        // Publish to RViz
        moveit_msgs::msg::DisplayRobotState msg;


        moveit::core::robotStateToRobotStateMsg(
            *robot_state_,
            msg.state);


        robot_state_pub_->publish(
            msg);


        // collision_detection::CollisionRequest collision_request;
        // collision_detection::CollisionResult collision_result;

        // collision_request.contacts = true;
        // collision_request.max_contacts = 100;

        // planning_scene->checkCollision(
        //     collision_request,
        //     collision_result
        // );

        // if (collision_result.collision)
        // {
        //     RCLCPP_WARN(get_logger(), "Collision detected!");

        //     for (const auto &contact_pair : collision_result.contacts)
        //     {
        //         const auto &link1 = contact_pair.first.first;
        //         const auto &link2 = contact_pair.first.second;

        //         RCLCPP_WARN(
        //             get_logger(),
        //             "Collision between: %s <--> %s",
        //             link1.c_str(),
        //             link2.c_str()
        //         );
        //     }
        // }

        return isStateValid(
            robot_state_.get());
    }


    // =========================================================================
    // COLLISION CHECK
    // =========================================================================

    bool isConfigurationValid(
        const std::vector<double>& joints)
    {
        return setRobotConfiguration(
            joints);
    }


    bool isStateValid(
        moveit::core::RobotState* state)
    {
        collision_detection::CollisionRequest req;
        collision_detection::CollisionResult res;


        req.contacts = false;
        req.distance = true;


        planning_scene_->checkCollision(
            req,
            res,
            *state);


        if (res.collision)
            return false;


        col_distance_ =
            res.distance;


        return true;
    }


    // =========================================================================
    // END-EFFECTOR WORKSPACE RESTRICTION
    // =========================================================================

    bool endEffectorInAllowedRegion(
        const std::vector<double>& joints)
    {
        robot_state_->setJointGroupPositions(
            joint_model_group_,
            joints);


        robot_state_->update();


        const auto* ee_link =
            robot_model_->getLinkModel(
                move_group_->getEndEffectorLink());


        if (!ee_link)
            return false;


        Eigen::Isometry3d T =
            robot_state_->getGlobalLinkTransform(
                ee_link);


        Eigen::Vector3d p =
            T.translation();


        // -----------------------------------------------------------------
        // CHANGE THESE TO YOUR ACTUAL EE REGION.
        // -----------------------------------------------------------------

        constexpr double x_min = -0.15; //0.15 is centered
        constexpr double x_max =  0.45;

        constexpr double y_min = -1.0;
        constexpr double y_max =  -0.25;

        constexpr double z_min =  0.15;
        constexpr double z_max =  1.5;


        return
            p.x() >= x_min &&
            p.x() <= x_max &&
            p.y() >= y_min &&
            p.y() <= y_max &&
            p.z() >= z_min &&
            p.z() <= z_max;
    }


    // =========================================================================
    // CANDIDATE EVALUATION
    // =========================================================================

    void evaluateCandidate(
        ConfigurationCandidate& candidate)
    {
        // -------------------------------------------------------------
        // Set candidate as current state.
        // -------------------------------------------------------------

        if (!setRobotConfiguration(
                candidate.joints))
        {
            candidate.valid = false;
            return;
        }


        // -------------------------------------------------------------
        // 1. Evaluate motion planning.
        // -------------------------------------------------------------

        candidate.planning_success_ratio =
            evaluateCandidatePlanning(
                candidate.joints);


        // -------------------------------------------------------------
        // 2. Compute metrics for this configuration.
        // -------------------------------------------------------------

        computeConfigurationMetrics(
            candidate);


        // -------------------------------------------------------------
        // 3. Calculate quality.
        //
        // Planning is deliberately dominant.
        // -------------------------------------------------------------

        constexpr double W_PLANNING = 10.0;

        constexpr double W_MANIPULABILITY = 1.0;

        constexpr double W_JOINT_LIMITS = 1.0;

        constexpr double W_COLLISION = 1.0;


        candidate.quality_score =
            W_PLANNING *
                candidate.planning_success_ratio

            +

            W_MANIPULABILITY *
                candidate.manipulability

            +

            W_JOINT_LIMITS *
                candidate.joint_centering_quality

            +

            W_COLLISION *
                candidate.collision_distance_quality;


        candidate.valid = true;
    }


    // =========================================================================
    // MOTION PLANNING FOR ONE CANDIDATE
    //
    // This is different from your previous evaluateReachability():
    //
    // There is NO IK starting configuration.
    //
    // candidate.joints is the actual starting configuration.
    //
    // The target poses are only used to evaluate how good this starting
    // configuration is.
    // =========================================================================

    double evaluateCandidatePlanning(
        const std::vector<double>& start_joints)
    {
        std::vector<geometry_msgs::msg::Pose>
            targets =
                getRepresentativeTargets();


        if (targets.empty())
        {
            RCLCPP_WARN(
                get_logger(),
                "No representative targets found");

            return 0.0;
        }


        // Set starting configuration.
        //
        // This is the important part.
        setRobotConfiguration(
            start_joints);


        int successful_plans = 0;


        for (const auto& target :
             targets)
        {
            move_group_->setStartState(
                *robot_state_);


            move_group_->setPoseTarget(
                target,
                move_group_->getEndEffectorLink());


            moveit::planning_interface::
                MoveGroupInterface::Plan plan;


            auto result =
                move_group_->plan(
                    plan);


            if (result ==
                moveit::core::MoveItErrorCode::SUCCESS)
            {
                ++successful_plans;
            }


            move_group_->clearPoseTargets();
        }


        return
            static_cast<double>(
                successful_plans)
            /
            static_cast<double>(
                targets.size());
    }


    // =========================================================================
    // REPRESENTATIVE TARGETS
    // =========================================================================

    std::vector<geometry_msgs::msg::Pose>
    getRepresentativeTargets()
    {
        std::vector<
            geometry_msgs::msg::Pose>
            targets;


        /*
         * OPTION 1:
         *
         * Use your existing pruning poses.
         *
         * You already have TF frames:
         *
         *     pruning_pose_0
         *     pruning_pose_1
         *     ...
         *
         * We take the first ~10.
         */


        const std::string prefix =
            "pruning_pose_";


        int total =
            getHighestPoseIndex(
                prefix) + 1;


        const int N =
            std::min(
                total,
                number_of_representative_targets_);


        for (int i = 0;
             i < N;
             ++i)
        {
            std::string frame =
                prefix +
                std::to_string(i);


            try
            {
                auto transform =
                    tf_buffer_.lookupTransform(
                        base_frame_,
                        frame,
                        tf2::TimePointZero);


                geometry_msgs::msg::Pose pose;


                pose.position.x =
                    transform.transform.translation.x;

                pose.position.y =
                    transform.transform.translation.y;

                pose.position.z =
                    transform.transform.translation.z;

                pose.orientation =
                    transform.transform.rotation;


                targets.push_back(
                    pose);
            }
            catch (
                tf2::TransformException& ex)
            {
                RCLCPP_WARN(
                    get_logger(),
                    "Could not find %s: %s",
                    frame.c_str(),
                    ex.what());
            }
        }


        return targets;
    }


    // =========================================================================
    // FIND HIGHEST TF INDEX
    // =========================================================================

    int getHighestPoseIndex(
        const std::string& prefix)
    {
        std::vector<std::string> frames;


        tf_buffer_._getFrameStrings(
            frames);


        int max_index = -1;


        for (const auto& frame :
             frames)
        {
            if (frame.rfind(prefix, 0) != 0)
                continue;


            std::string suffix =
                frame.substr(
                    prefix.size());


            try
            {
                int idx =
                    std::stoi(suffix);


                max_index =
                    std::max(
                        max_index,
                        idx);
            }
            catch (...)
            {
            }
        }


        return max_index;
    }


    // =========================================================================
    // COMPUTE CONFIGURATION METRICS
    // =========================================================================

    void computeConfigurationMetrics(
        ConfigurationCandidate& candidate)
    {
        robot_state_->setJointGroupPositions(
            joint_model_group_,
            candidate.joints);


        robot_state_->update();


        // -------------------------------------------------------------
        // Joint limits
        // -------------------------------------------------------------

        candidate.joint_centering_quality =
            computeJointCenteringQuality();


        // -------------------------------------------------------------
        // Collision distance
        // -------------------------------------------------------------

        candidate.collision_distance_quality =
            computeCollisionDistanceQuality();


        // -------------------------------------------------------------
        // Collision interference
        // -------------------------------------------------------------

        // candidate.collision_quality =
        //     1.0 -
        //     computeCollisionCost();


        // -------------------------------------------------------------
        // Jacobian
        // -------------------------------------------------------------

        Eigen::MatrixXd J;


        const auto* ee_link =
            robot_model_->getLinkModel(
                move_group_->getEndEffectorLink());


        if (!ee_link)
        {
            candidate.manipulability = 0.0;
            // candidate.manipulability_world_y = 0.0;
            // candidate.manipulability_tool_neg_z = 0.0;

            return;
        }


        robot_state_->getJacobian(
            joint_model_group_,
            ee_link,
            Eigen::Vector3d::Zero(),
            J);


        if (J.rows() != 6 ||
            J.cols() == 0)
        {
            candidate.manipulability = 0.0;
            // candidate.manipulability_world_y = 0.0;
            // candidate.manipulability_tool_neg_z = 0.0;

            return;
        }


        Eigen::MatrixXd JJt =
            J * J.transpose();


        // -------------------------------------------------------------
        // Yoshikawa manipulability
        // -------------------------------------------------------------

        double determinant =
            JJt.determinant();


        if (determinant > 0.0)
        {
            double w =
                std::sqrt(
                    determinant);


            double w_scaled =
                w /
                std::pow(
                    robot_length_,
                    3);


            if (normalization_loaded_)
            {
                candidate.manipulability =
                    1.0 -
                    std::exp(
                        -k_yosh_ *
                        w_scaled);
            }
            else
            {
                candidate.manipulability =
                    w_scaled;
            }
        }
        else
        {
            candidate.manipulability =
                0.0;
        }
    }


    // =========================================================================
    // JOINT CENTERING
    // =========================================================================

    double computeJointCenteringQuality()
    {
        const auto& joints =
            joint_model_group_->getActiveJointModels();


        if (joints.empty())
            return 0.0;


        double cost = 0.0;


        for (size_t i = 0;
             i < joints.size();
             ++i)
        {
            const auto& bounds =
                joints[i]->getVariableBounds()[0];


            double q =
                robot_state_->getVariablePosition(
                    joints[i]->getFirstVariableIndex());


            double q_min =
                bounds.min_position_;


            double q_max =
                bounds.max_position_;


            double q_mid =
                0.5 *
                (q_min + q_max);


            double range =
                q_max - q_min;


            if (range <= 0.0)
                continue;


            double normalized =
                2.0 *
                (q - q_mid) /
                range;


            cost +=
                normalized *
                normalized;
        }


        cost /=
            static_cast<double>(
                joints.size());


        return
            1.0 - cost;
    }


    // =========================================================================
    // COLLISION DISTANCE QUALITY
    // =========================================================================

    double computeCollisionDistanceQuality()
    {
        if (col_distance_ >=
            threshold_col)
        {
            return 1.0;
        }


        if (col_distance_ <= 0.0)
            return 0.0;


        double d =
            col_distance_;


        double cost =
            std::exp(
                -(d /
                (threshold_col - d)));


        return
            1.0 - cost;
    }


    // =========================================================================
    // COLLISION INTERFERENCE
    // =========================================================================

    std::vector<geometry_msgs::msg::Point>
    interpolatePoints(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        int samples)
    {
        std::vector<
            geometry_msgs::msg::Point>
            points;


        for (int i = 0;
             i <= samples;
             ++i)
        {
            double t =
                static_cast<double>(i) /
                static_cast<double>(samples);


            Eigen::Vector3d p =
                (1.0 - t) * p1 +
                t * p2;


            geometry_msgs::msg::Point msg;


            msg.x = p.x();
            msg.y = p.y();
            msg.z = p.z();


            points.push_back(
                msg);
        }


        return points;
    }


    double computeCollisionCost()
    {
        std::vector<
            geometry_msgs::msg::Point>
            sampled_points;


        for (const auto& pair :
             col_link_pairs)
        {
            Eigen::Isometry3d T1_world =
                robot_state_->getGlobalLinkTransform(
                    pair.first);


            Eigen::Isometry3d T2_world =
                robot_state_->getGlobalLinkTransform(
                    pair.second);


            Eigen::Isometry3d T1_vineyard =
                T_vineyard_world *
                T1_world;


            Eigen::Isometry3d T2_vineyard =
                T_vineyard_world *
                T2_world;


            Eigen::Vector3d p1 =
                T1_vineyard.translation();


            Eigen::Vector3d p2 =
                T2_vineyard.translation();


            auto points =
                interpolatePoints(
                    p1,
                    p2,
                    10);


            sampled_points.insert(
                sampled_points.end(),
                points.begin(),
                points.end());
        }


        auto request =
            std::make_shared<
                custom_interfaces::srv::
                CollisionCost::Request>();


        request->points =
            sampled_points;


        auto future =
            col_cost_client_->async_send_request(
                request);


        auto status =
            future.wait_for(
                std::chrono::milliseconds(500));


        if (status !=
            std::future_status::ready)
        {
            RCLCPP_WARN(
                get_logger(),
                "Collision service timeout");


            return 1.0;
        }


        auto response =
            future.get();


        return response->cost;
    }


    // =========================================================================
    // SPATIAL SMOOTHING
    // =========================================================================
    //
    // We DO NOT average joint configurations.
    //
    // Instead, at each grid point we choose one of the already evaluated
    // candidates.
    //
    // This is important because q_front and q_back should not be averaged.
    // =========================================================================

    void smoothConfigurationTable(
        std::vector<PlacementConfig>& table,
        int iterations,
        double smoothness_weight)
    {
        // -------------------------------------------------------------
        // Initial state = locally best candidate.
        // -------------------------------------------------------------

        for (auto& point :
             table)
        {
            if (!point.candidates.empty())
            {
                point.selected_candidate =
                    0;
            }
        }


        // -------------------------------------------------------------
        // Iterative coordinate descent.
        // -------------------------------------------------------------

        for (int iteration = 0;
             iteration < iterations;
             ++iteration)
        {
            bool changed = false;


            for (size_t i = 0;
                 i < table.size();
                 ++i)
            {
                if (table[i].candidates.empty())
                    continue;


                int current =
                    table[i].selected_candidate;


                int best =
                    current;


                double best_energy =
                    candidateEnergy(
                        table,
                        i,
                        current,
                        smoothness_weight);


                // -----------------------------------------------------
                // Try every stored candidate.
                // -----------------------------------------------------

                for (size_t c = 0;
                     c < table[i].candidates.size();
                     ++c)
                {
                    double energy =
                        candidateEnergy(
                            table,
                            i,
                            static_cast<int>(c),
                            smoothness_weight);


                    if (energy < best_energy)
                    {
                        best_energy =
                            energy;

                        best =
                            static_cast<int>(c);
                    }
                }


                if (best != current)
                {
                    table[i].selected_candidate =
                        best;

                    changed = true;
                }
            }


            RCLCPP_INFO(
                get_logger(),
                "Smoothing iteration %d: changed=%s",
                iteration + 1,
                changed ? "true" : "false");


            if (!changed)
                break;
        }
    }


    // =========================================================================
    // CANDIDATE ENERGY
    // =========================================================================

    double candidateEnergy(
        const std::vector<PlacementConfig>& table,
        size_t index,
        int candidate_index,
        double smoothness_weight)
    {
        if (candidate_index < 0 ||
            candidate_index >=
                static_cast<int>(
                    table[index].candidates.size()))
        {
            return
                std::numeric_limits<double>::infinity();
        }


        const auto& candidate =
            table[index].candidates[
                candidate_index];


        // -------------------------------------------------------------
        // Higher quality = lower energy.
        // -------------------------------------------------------------

        double energy =
            -candidate.quality_score;


        // -------------------------------------------------------------
        // Six grid neighbours:
        //
        // Y +/- 1
        // Z +/- 1
        // Roll +/- 1
        // -------------------------------------------------------------

        std::vector<size_t> neighbors =
            getGridNeighbors(
                table,
                index); //Works for 3 dim or less


        for (size_t neighbour :
             neighbors)
        {
            if (table[neighbour].selected_candidate < 0)
                continue;


            int neighbour_candidate_index =
                table[neighbour].selected_candidate;


            if (neighbour_candidate_index >=
                static_cast<int>(
                    table[neighbour].candidates.size()))
            {
                continue;
            }


            const auto& neighbour_candidate =
                table[neighbour].candidates[
                    neighbour_candidate_index];


            double distance =
                jointConfigurationDistance(
                    candidate.joints,
                    neighbour_candidate.joints);


            energy +=
                smoothness_weight *
                distance;
        }


        return energy;
    }


    // =========================================================================
    // GRID NEIGHBOURS
    // =========================================================================

    std::vector<size_t> getGridNeighbors(
        const std::vector<PlacementConfig>& table,
        size_t index)
    {
        std::vector<size_t> neighbours;


        const Placement& p =
            table[index].placement;


        const int dy[] =
            {-1, 1, 0, 0, 0, 0};


        const int dz[] =
            {0, 0, -1, 1, 0, 0};


        const int dr[] =
            {0, 0, 0, 0, -1, 1};


        for (int n = 0;
             n < 6;
             ++n)
        {
            int iy =
                p.iy + dy[n];


            int iz =
                p.iz + dz[n];


            int ir =
                p.iroll + dr[n];


            for (size_t i = 0;
                 i < table.size();
                 ++i)
            {
                const Placement& other =
                    table[i].placement;


                if (other.iy == iy &&
                    other.iz == iz &&
                    other.iroll == ir)
                {
                    neighbours.push_back(i);
                    break;
                }
            }
        }


        return neighbours;
    }


    // =========================================================================
    // JOINT CONFIGURATION DISTANCE
    // =========================================================================

    double jointConfigurationDistance(
        const std::vector<double>& q1,
        const std::vector<double>& q2)
    {
        if (q1.size() != q2.size())
        {
            return
                std::numeric_limits<double>::infinity();
        }


        double distance = 0.0;


        for (size_t i = 0;
             i < q1.size();
             ++i)
        {
            double diff =
                q1[i] - q2[i];


            distance +=
                diff * diff;
        }


        return
            std::sqrt(distance);
    }


    // =========================================================================
    // PRINT FINAL TABLE
    // =========================================================================

    void printFinalTable(
        const std::vector<PlacementConfig>& table)
    {
        RCLCPP_INFO(
            get_logger(),
            "==============================================");

        RCLCPP_INFO(
            get_logger(),
            "FINAL SMOOTHED CONFIGURATION TABLE");

        RCLCPP_INFO(
            get_logger(),
            "==============================================");


        for (const auto& point :
             table)
        {
            if (point.selected_candidate < 0)
                continue;


            const auto& candidate =
                point.candidates[
                    point.selected_candidate];


            RCLCPP_INFO(
                get_logger(),
                "Y=%.4f Z=%.4f Roll=%.4f | "
                "candidate=%d | "
                "quality=%.4f | "
                "planning=%.3f | "
                "q=[%.3f %.3f %.3f %.3f %.3f %.3f]",
                point.placement.values[0],
                point.placement.values[1],
                point.placement.values[2],
                point.selected_candidate,
                candidate.quality_score,
                candidate.planning_success_ratio,
                candidate.joints[0],
                candidate.joints[1],
                candidate.joints[2],
                candidate.joints[3],
                candidate.joints[4],
                candidate.joints[5]);
        }
    }


    // =========================================================================
    // SAVE TABLE
    // =========================================================================

    void saveRawConfigurationTable(
        const std::vector<PlacementConfig>& table)
    {
        std::string path =
            "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/raw_configuration_table_2.txt";

        std::ofstream file(path);

        if (!file.is_open())
        {
            RCLCPP_ERROR(
                get_logger(),
                "Could not open %s",
                path.c_str());

            return;
        }


        // Header
        file << "# "
            << "iy iz iroll "
            << "Y Z Roll "
            << "candidate "
            << "quality "
            << "q0 q1 q2 q3 q4 q5\n";


        for (const auto& point : table)
        {
            for (size_t c = 0;
                c < point.candidates.size();
                ++c)
            {
                const auto& candidate =
                    point.candidates[c];


                file
                    << point.placement.iy
                    << " "
                    << point.placement.iz
                    << " "
                    << point.placement.iroll
                    << " "

                    << point.placement.values[0]
                    << " "
                    << point.placement.values[1]
                    << " "
                    << point.placement.values[2]
                    << " "

                    << c
                    << " "

                    << candidate.quality_score
                    << " "

                    << candidate.planning_success_ratio;


                for (double q : candidate.joints)
                {
                    file << " " << q;
                }

                file << "\n";
            }
        }


        file.close();


        RCLCPP_INFO(
            get_logger(),
            "Saved RAW configuration table to %s",
            path.c_str());
    }

    void saveConfigurationTable(
        const std::vector<PlacementConfig>& table)
    {
        std::string path =
            "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/initial_configuration_table_2.txt";


        std::ofstream file(
            path);


        if (!file.is_open())
        {
            RCLCPP_ERROR(
                get_logger(),
                "Could not open %s",
                path.c_str());

            return;
        }


        file << "# Y Z ROLL "
             << "candidate "
             << "quality "
             << "planning "
             << "q0 q1 q2 q3 q4 q5\n";


        for (const auto& point :
             table)
        {
            if (point.selected_candidate < 0)
                continue;


            const auto& candidate =
                point.candidates[
                    point.selected_candidate];


            file
                << point.placement.values[0]
                << " "
                << point.placement.values[1]
                << " "
                << point.placement.values[2]
                << " "
                << point.selected_candidate
                << " "
                << candidate.quality_score
                << " "
                << candidate.planning_success_ratio;


            for (double q :
                 candidate.joints)
            {
                file << " " << q;
            }


            file << "\n";
        }


        file.close();


        RCLCPP_INFO(
            get_logger(),
            "Saved final configuration table to %s",
            path.c_str());
    }


    // =========================================================================
    // NORMALIZATION
    //
    // These are kept from your existing implementation.
    // =========================================================================

    bool loadNormalization()
    {
        if (!std::filesystem::exists(
                yaml_src_path_))
        {
            RCLCPP_WARN(
                get_logger(),
                "Normalization file not found");

            return false;
        }


        YAML::Node config =
            YAML::LoadFile(
                yaml_src_path_);


        k_yosh_ =
            config["manipulability"]
                ["k_yosh"]
                .as<double>();


        RCLCPP_INFO(
            get_logger(),
            "Loaded normalization constants");


        return true;
    }

    void addFixedBoxObstacle()
    {
        moveit_msgs::msg::CollisionObject obj;
        obj.id = "temp_box";
        obj.header.frame_id = "world";
        obj.pose.position.x = 0.15;
        obj.pose.position.y = -1.0;
        float dim_z = 0.5;
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

    void removeFixedBoxObstacle()
    {
        moveit_msgs::msg::CollisionObject obj;
        obj.id = "temp_box";
        obj.header.frame_id = "world";
        obj.operation = obj.REMOVE;

        planning_scene_->processCollisionObjectMsg(obj);

        // RCLCPP_INFO(get_logger(), "Obstacle removed");
    }


    // =========================================================================
    // MEMBERS
    // =========================================================================

    std::string robot_type_ =
        "ur5e";


    std::string tool_placement_ =
        "Lshape";


    moveit::core::RobotModelPtr
        robot_model_;


    moveit::core::RobotStatePtr
        robot_state_;


    planning_scene::PlanningScenePtr
        planning_scene_;


    const moveit::core::JointModelGroup*
        joint_model_group_;


    std::shared_ptr<
        moveit::planning_interface::
        MoveGroupInterface>
        move_group_;


    rclcpp::Publisher<
        moveit_msgs::msg::DisplayRobotState>::SharedPtr
        robot_state_pub_;


    rclcpp::Publisher<
        sensor_msgs::msg::JointState>::SharedPtr
        publisher_passive_joints_;


    rclcpp::Client<
        custom_interfaces::srv::CollisionCost>::SharedPtr
        col_cost_client_;


    tf2_ros::Buffer
        tf_buffer_;


    tf2_ros::TransformListener
        tf_listener_;


    std::mt19937 rng_;


    Eigen::Isometry3d
        T_vineyard_world;


    std::string
        base_frame_;


    double
        col_distance_ = 0.0;


    std::vector<
        std::pair<std::string, std::string>>
        col_link_pairs =
    {
        {"forearm_link", "wrist_1_link"},
        {"wrist_1_link", "wrist_2_link"},
        {"wrist_2_link", "wrist_3_link"},
        {"wrist_3_link", "tool0"},
        {"tool0", "ee_link"}
    };


    double
        k_yosh_ = -1.0;


    bool
        normalization_loaded_ = false;


    double
        robot_length_ = 0.85;


    double
        threshold_col = 0.15;


    int
        number_of_representative_targets_ = 10;


    std::string
        yaml_src_path_ =
        "/home/rosdev/ros2_ws/src/"
        "moveit_cpp_demo/config/"
        "manipulability_ur5e.yaml";
};


// ============================================================================
// MAIN
// ============================================================================

int main(
    int argc,
    char** argv)
{
    rclcpp::init(
        argc,
        argv);


    auto node =
        std::make_shared<
            IKReachabilityNode>();


    std::thread init_thread(
        [node]()
        {
            node->initialize();
        });


    rclcpp::spin(
        node);


    init_thread.join();


    rclcpp::shutdown();


    return 0;
}