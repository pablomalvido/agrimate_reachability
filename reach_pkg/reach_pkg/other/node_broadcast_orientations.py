import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from custom_interfaces.srv import SampleOrientations


class OrientationTFBroadcaster(Node):

    def __init__(self):
        super().__init__('orientation_tf_broadcaster')

        # TF broadcaster
        self.br = TransformBroadcaster(self)

        # Service client
        self.cli = self.create_client(
            SampleOrientations,
            'sample_orientations'
        )

        self.get_logger().info("Waiting for sampling service...")
        self.cli.wait_for_service()

        # Request n orientations
        self.request_orientations()

        # Timer to continuously broadcast TFs
        self.timer = self.create_timer(0.1, self.broadcast_transforms)

        self.transforms = []

    def request_orientations(self):
        req = SampleOrientations.Request()
        req.n = 60

        # Example cone limits (radians)
        req.max_angle_y = 1.05 # ~60 degrees
        req.max_angle_z = 1.05 # 60 degrees
        req.max_roll = 1.05 # 60 degrees

        future = self.cli.call_async(req)
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info("Received orientations")

            self.transforms = []

            for i, q in enumerate(response.orientations):
                t = TransformStamped()

                t.header.frame_id = "map"
                t.child_frame_id = f"sample_pose_{i}"

                # fixed position
                t.transform.translation.x = 1.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0

                # orientation from service
                t.transform.rotation = q

                self.transforms.append(t)

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def broadcast_transforms(self):
        if not self.transforms:
            return

        now = self.get_clock().now().to_msg()

        for t in self.transforms:
            t.header.stamp = now

        self.br.sendTransform(self.transforms)


def main(args=None):
    rclpy.init(args=args)
    node = OrientationTFBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()