#pragma once

#include <vector>

struct PlacementSample
{
    double y;
    double z;
    double roll;
    double score;
    int iter;
};

struct Placement
{
    double y;
    double z;
    double roll;
};

struct SearchBounds
{
    double y_min;
    double y_max;

    double z_min;
    double z_max;

    double roll_min;
    double roll_max;
};

class SimplifiedBayesianOptimizer
{
public:

    SimplifiedBayesianOptimizer(
        double ymin,
        double ymax,
        double zmin,
        double zmax,
        double roll_min,
        double roll_max);

    Placement proposeNext(
        const std::vector<PlacementSample>& history);
        
private:

    double ymin_;
    double ymax_;

    double zmin_;
    double zmax_;

    double roll_min_;
    double roll_max_;

    SearchBounds bounds_;
};