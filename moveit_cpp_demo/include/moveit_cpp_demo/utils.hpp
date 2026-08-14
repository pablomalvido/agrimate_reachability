#pragma once

#include <string>
#include <vector>
#include "moveit_cpp_demo/types.hpp"

// Replace HistoryEntry with the actual type of your history elements.
std::string saveHistoryToFile(
    const std::string &title,
    const std::vector<FullPlacementSample> &history
);