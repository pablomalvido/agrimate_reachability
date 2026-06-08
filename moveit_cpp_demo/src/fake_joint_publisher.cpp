#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <random>
#include <vector>
#include <algorithm>

class RandomJointPublisher : public rclcpp::Node
{
public:
  RandomJointPublisher()
  : Node("random_joint_publisher"),
    rng_(std::random_device{}())
  {
    joint_pub_ =
      create_publisher<sensor_msgs::msg::JointState>(
        "/joint_states", 10);

    timer_ =
      create_wall_timer(
        std::chrono::seconds(2),
        std::bind(
          &RandomJointPublisher::timerCallback,
          this));

    RCLCPP_INFO(
      get_logger(),
      "Random joint publisher started");
  }

private:

  //--------------------------------------------------
  // YOUR MAPPING HERE
  //--------------------------------------------------

  std::vector<double> computeRobotConfig(
    double slider,
    double cylinder)
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

    double alpha = (slider - slider_min) / (slider_max - slider_min);

    alpha = std::clamp(alpha, 0.0, 1.0);

    q[1] =  -1.35 + alpha * 1.35;
    q[1] += -cylinder * 1.05;

    return q;
  }

  //--------------------------------------------------
  // Timer
  //--------------------------------------------------

  void timerCallback()
  {
    //--------------------------------------------------
    // Random passive joints
    //--------------------------------------------------

    std::uniform_real_distribution<double>
      slider_dist(slider_min, slider_max);

    std::uniform_real_distribution<double>
      cylinder_dist(cyl_min, cyl_max);

    double slider =
      slider_dist(rng_);

    double cylinder =
      cylinder_dist(rng_);

    //--------------------------------------------------
    // Compute robot configuration
    //--------------------------------------------------

    std::vector<double> q =
      computeRobotConfig(
        slider,
        cylinder);

    //--------------------------------------------------
    // Publish all joints
    //--------------------------------------------------

    sensor_msgs::msg::JointState msg;

    msg.header.stamp =
      now();

    msg.name =
    {
      //------------------------------------------------
      // Passive joints
      //------------------------------------------------
      "platform_to_slider",
      "platform_to_cylinder",

      //------------------------------------------------
      // UR joints
      //------------------------------------------------
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
    };

    msg.position =
    {
      slider,
      cylinder,

      q[0],
      q[1],
      q[2],
      q[3],
      q[4],
      q[5]
    };

    joint_pub_->publish(msg);

    RCLCPP_INFO(
      get_logger(),
      "Published slider=%.3f cylinder=%.3f",
      slider,
      cylinder);
  }

  //--------------------------------------------------
  // Members
  //--------------------------------------------------

  rclcpp::Publisher<
    sensor_msgs::msg::JointState>::SharedPtr joint_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::mt19937 rng_;

  double cyl_min = -0.8; //rad
  double cyl_max = 0.8; //rad
  double slider_min = 0.0; //m
  double slider_max = 0.3; //m
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<RandomJointPublisher>();

  rclcpp::spin(node);

  rclcpp::shutdown();

  return 0;
}