#include <rclcpp/rclcpp.hpp>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <moveit/collision_detection/collision_common.h>


#include "custom_interfaces/srv/add_box.hpp"
#include "custom_interfaces/srv/remove_box.hpp"

class PlanningSceneServiceNode : public rclcpp::Node
{
public:
  PlanningSceneServiceNode()
  : Node("planning_scene_service_node")
  {
    add_service_ = this->create_service<custom_interfaces::srv::AddBox>(
      "add_box",
      std::bind(&PlanningSceneServiceNode::addBoxCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    remove_service_ = this->create_service<custom_interfaces::srv::RemoveBox>(
      "remove_box",
      std::bind(&PlanningSceneServiceNode::removeBoxCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Planning Scene Service Node ready.");
  }

private:

  void addBoxCallback(
    const std::shared_ptr<custom_interfaces::srv::AddBox::Request> request,
    std::shared_ptr<custom_interfaces::srv::AddBox::Response> response)
  {
    moveit_msgs::msg::CollisionObject obj;

    obj.id = request->id;
    obj.header.frame_id = "world";  // ⚠️ adjust if needed

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {
      request->size_x,
      request->size_y,
      request->size_z
    };

    obj.primitives.push_back(primitive);
    obj.primitive_poses.push_back(request->pose);
    obj.operation = obj.ADD;

    planning_scene_interface_.applyCollisionObject(obj);

    response->success = true;
    response->message = "Box added: " + request->id;

    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  }

  void removeBoxCallback(
    const std::shared_ptr<custom_interfaces::srv::RemoveBox::Request> request,
    std::shared_ptr<custom_interfaces::srv::RemoveBox::Response> response)
  {
    std::vector<std::string> ids;
    ids.push_back(request->id);

    planning_scene_interface_.removeCollisionObjects(ids);

    response->success = true;
    response->message = "Box removed: " + request->id;

    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  }

  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  rclcpp::Service<custom_interfaces::srv::AddBox>::SharedPtr add_service_;
  rclcpp::Service<custom_interfaces::srv::RemoveBox>::SharedPtr remove_service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PlanningSceneServiceNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}