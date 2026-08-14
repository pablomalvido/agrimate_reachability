#include "moveit_cpp_demo/simplified_bayesian_optimizer.hpp"

#include <random>
#include <limits>
#include <cmath>

static Placement normalizePlacement(
    const Placement& p,
    const SearchBounds& b);

static double normalizedDistance(
    const Placement& a,
    const Placement& b,
    const SearchBounds& bounds);

SimplifiedBayesianOptimizer::SimplifiedBayesianOptimizer(
    double ymin,
    double ymax,
    double zmin,
    double zmax,
    double roll_min,
    double roll_max)
: ymin_(ymin)
, ymax_(ymax)
, zmin_(zmin)
, zmax_(zmax)
, roll_min_(roll_min)
, roll_max_(roll_max)
, bounds_{
    ymin,
    ymax,
    zmin,
    zmax,
    roll_min,
    roll_max
  }
{
}

Placement SimplifiedBayesianOptimizer::proposeNext(
    const std::vector<PlacementSample>& history)
{
    static std::mt19937 rng(std::random_device{}());

    std::uniform_real_distribution<double> dy(ymin_, ymax_);
    std::uniform_real_distribution<double> dz(zmin_, zmax_);
    std::uniform_real_distribution<double> dr(
        roll_min_,
        roll_max_);

    //--------------------------------------------------
    // Initial exploration
    //--------------------------------------------------

    if (history.size() < 10)
    {
        return {
            dy(rng),
            dz(rng),
            dr(rng)
        };
    }

    //--------------------------------------------------
    // Current best
    //--------------------------------------------------

    double best_score =
        -std::numeric_limits<double>::infinity();

    for (const auto& s : history)
    {
        best_score =
            std::max(best_score, s.score);
    }

    //--------------------------------------------------
    // Search candidates
    //--------------------------------------------------

    double best_acquisition =
        -std::numeric_limits<double>::infinity();

    Placement best_candidate;

    constexpr int NUM_CANDIDATES = 1000;

    for (int i = 0; i < NUM_CANDIDATES; ++i)
    {
        Placement p{
            dy(rng),
            dz(rng),
            dr(rng)
        };

        //------------------------------------------------
        // Surrogate prediction
        //------------------------------------------------

        double weighted_sum = 0.0;
        double weight_total = 0.0;

        double nearest_dist =
            std::numeric_limits<double>::max();

        for (const auto& s : history)
        {
            Placement s_p{s.y, s.z, s.roll};
            double dist = normalizedDistance(p, s_p, bounds_);

            nearest_dist = std::min(nearest_dist, dist);

            double w = 1.0 / (dist + 1e-6);

            weighted_sum += w * s.score;
            weight_total += w;
        }

        double mean = weighted_sum / weight_total;

        //------------------------------------------------
        // Uncertainty estimate
        //------------------------------------------------

        double sigma = nearest_dist;

        //------------------------------------------------
        // Expected-improvement-like score
        //------------------------------------------------

        double acquisition =
            (mean - best_score)
            + 0.5 * sigma;

        if (acquisition > best_acquisition)
        {
            best_acquisition = acquisition;
            best_candidate = p;
        }
    }

    return best_candidate;
}


Placement normalizePlacement(
    const Placement& p,
    const SearchBounds& b)
{
    Placement n;

    n.y =
      (p.y - b.y_min) /
      (b.y_max - b.y_min);

    n.z =
      (p.z - b.z_min) /
      (b.z_max - b.z_min);

    n.roll =
      (p.roll - b.roll_min) /
      (b.roll_max - b.roll_min);

    return n;
}

double normalizedDistance(
    const Placement& a,
    const Placement& b,
    const SearchBounds& bounds)
{
    Placement na =
      normalizePlacement(a, bounds);

    Placement nb =
      normalizePlacement(b, bounds);

    double dy = na.y - nb.y;
    double dz = na.z - nb.z;
    double dr = na.roll - nb.roll;

    return std::sqrt(
      dy*dy +
      dz*dz +
      dr*dr);
}