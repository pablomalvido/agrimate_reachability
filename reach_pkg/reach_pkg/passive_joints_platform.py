import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class PassiveJointPublisher(Node):
    def __init__(self):
        super().__init__('passive_joint_pub')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Subscriber receiving desired values
        self.sub = self.create_subscription(
            JointState,
            '/passive_joint_commands',
            self.command_callback,
            10)

        
        self.position_slider = 0.0  # <-- change this dynamically if you want
        self.position_cylinder = 0.0  # <-- change this dynamically if you want
        self.position_vineyard = -0.7  # <-- change this dynamically if you want
        self.position_slider_horizontal = 0.0  # <-- change this dynamically if you want
        self.timer = self.create_timer(0.1, self.publish_joint)

    def publish_joint(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['platform_to_slider', 'platform_to_cylinder', 'world_to_vineyard', 'horizontal_slider']
        msg.position = [self.position_slider, self.position_cylinder, self.position_vineyard, self.position_slider_horizontal]
        self.pub.publish(msg)

    def command_callback(self, msg):
        #self.get_logger().info(str(msg))
        for name, pos in zip(msg.name, msg.position):
            if name == "platform_to_slider":
                self.position_slider = pos

            elif name == "platform_to_cylinder":
                self.position_cylinder = pos

            elif name == "world_to_vineyard":
                self.position_vineyard = pos

            elif name == "horizontal_slider":
                self.position_slider_horizontal = pos

        self.publish_joint() #Fast update

        self.get_logger().info(f"Received slider={self.position_slider:.3f}, "
            f"cylinder={self.position_cylinder:.3f}, "
            f"vineyard={self.position_vineyard:.3f}, "
            f"horizontal_slider={self.position_slider_horizontal:.3f}")


def main(args=None):

    rclpy.init(args=args)

    node = PassiveJointPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()