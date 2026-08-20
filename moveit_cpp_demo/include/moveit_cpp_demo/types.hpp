#pragma once

#include <string>
#include <vector>

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
    double y_offset_obstacle = 0.0; // Offset for the obstacle in the Y direction
    std::string obstacle = "";
};

struct ReachabilityResult
{
    int ik_success_count = 0;
    int ik_total_count = 0;
    int planning_success_count = 0;
    int planning_total_count = 0;
    double ik_success_ratio = 0.0;
    double planning_success_ratio = 0.0;

    MeanMetrics mean_metrics;

    double total_cost = 0.0;

    std::vector<IKMetrics> metrics_list;
};

struct FullPlacementSample
{
    double y;
    double z;
    double roll;
    ReachabilityResult result1;
    ReachabilityResult result2;
    ReachabilityResult result3;
    double score;
    int iter;
};