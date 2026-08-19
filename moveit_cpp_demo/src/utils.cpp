#include "moveit_cpp_demo/types.hpp"
#include "moveit_cpp_demo/utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace fs = std::filesystem;

const std::string package_path =
    ament_index_cpp::get_package_share_directory("moveit_cpp_demo");

const fs::path data_dir = DATA_DIR;

std::string saveHistoryToFile(
    const std::string &title,
    const std::vector<FullPlacementSample> &history)
{

    fs::create_directories(data_dir);

    int highest_used_index = 0;

    const std::regex pattern(
        "^" + title + R"(_(\d+)\.txt$)"
    );

    // Find highest existing index
    for (const auto &entry : fs::directory_iterator(data_dir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::string filename =
            entry.path().filename().string();

        std::smatch match;

        if (std::regex_match(filename, match, pattern))
        {
            const int index = std::stoi(match[1].str());

            highest_used_index =
                std::max(highest_used_index, index);
        }
    }

    const int new_index = highest_used_index + 1;

    const fs::path filepath = data_dir / "scores" / (title + "_" + std::to_string(new_index) + ".txt");

    std::ofstream file(filepath);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: "
                  << filepath
                  << std::endl;

        return "";
    }

    std::vector<FullPlacementSample> sorted_history = history;

    std::sort(
        sorted_history.begin(),
        sorted_history.end(),
        [](const auto &a, const auto &b)
        {
            return a.score > b.score;
        }
    );

    // Save history
    for (size_t i = 0;
         i < std::min<size_t>(1000, sorted_history.size());
         ++i)
    {
        const auto &h = sorted_history[i];

        file << "Rank " << i + 1
             << " | Iteration=" << h.iter
             << " | y=" << h.y
             << " z=" << h.z
             << " roll=" << h.roll
             << " score=" << h.score
             << " score_prune=" << h.result1.total_cost
             << " score_scan=" << h.result2.total_cost
             << " score_grasp=" << h.result3.total_cost
             << " prune_ik_success_ratio=" << h.result1.ik_success_ratio
             << " prune_planning_success_ratio=" << h.result1.planning_success_ratio
             << " prune_joint_centering_cost=" << h.result1.mean_metrics.joint_centering_cost
             << " prune_col_distance=" << h.result1.mean_metrics.col_distance_norm
             << " prune_col_interference=" << h.result1.mean_metrics.col_interference
             << " prune_manipulability=" << h.result1.mean_metrics.manipulability_norm
             << " prune_manip_world_y=" << h.result1.mean_metrics.manip_world_y_norm
             << " prune_manip_tool_neg_z=" << h.result1.mean_metrics.manip_tool_neg_z_norm
             << " scan_ik_success_ratio=" << h.result2.ik_success_ratio
             << " scan_planning_success_ratio=" << h.result2.planning_success_ratio
             << " scan_joint_centering_cost=" << h.result2.mean_metrics.joint_centering_cost
             << " scan_col_distance=" << h.result2.mean_metrics.col_distance_norm
             << " scan_col_interference=" << h.result2.mean_metrics.col_interference
             << " scan_manipulability=" << h.result2.mean_metrics.manipulability_norm
             << " scan_manip_world_y=" << h.result2.mean_metrics.manip_world_y_norm
             << " scan_manip_tool_neg_z=" << h.result2.mean_metrics.manip_tool_neg_z_norm
             << " grasp_ik_success_ratio=" << h.result3.ik_success_ratio
             << " grasp_planning_success_ratio=" << h.result3.planning_success_ratio
             << " grasp_joint_centering_cost=" << h.result3.mean_metrics.joint_centering_cost
             << " grasp_col_distance=" << h.result3.mean_metrics.col_distance_norm
             << " grasp_col_interference=" << h.result3.mean_metrics.col_interference
             << " grasp_manipulability=" << h.result3.mean_metrics.manipulability_norm
             << " grasp_manip_world_y=" << h.result3.mean_metrics.manip_world_y_norm
             << " grasp_manip_tool_neg_z=" << h.result3.mean_metrics.manip_tool_neg_z_norm
             << '\n';
    }

    file.close();

    return filepath;
}