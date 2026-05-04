import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class PassiveJointPublisher(Node):
    def __init__(self):
        super().__init__('passive_joint_pub')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.position_slider = 0.0  # <-- change this dynamically if you want
        self.position_cylinder = 0.0  # <-- change this dynamically if you want
        self.timer = self.create_timer(0.1, self.publish_joint)

    def publish_joint(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['platform_to_slider', 'platform_to_cylinder']
        msg.position = [self.position_slider, self.position_cylinder]
        self.pub.publish(msg)

rclpy.init()
node = PassiveJointPublisher()
rclpy.spin(node)