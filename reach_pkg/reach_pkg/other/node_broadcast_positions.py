#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from custom_interfaces.srv import SamplePoints

import tf2_ros


class SampleTFBroadcaster(Node):

    def __init__(self):
        super().__init__('sample_tf_broadcaster')

        # TF broadcaster
        self.br = tf2_ros.TransformBroadcaster(self)

        # Service client
        self.cli = self.create_client(SamplePoints, 'sample_points')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for sample_points service...')

        self.get_logger().info('Service available, requesting samples')

        self.request_samples()

    # =========================
    # SERVICE CALL
    # =========================
    def request_samples(self):

        req = SamplePoints.Request()
        req.n = 25

        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self.response_callback)

    # =========================
    # RESPONSE CALLBACK
    # =========================
    def response_callback(self, future):

        try:
            response = future.result()
            self.get_logger().info(f"Received {len(response.points)} points")

            self.broadcast_tfs(response.points)

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    # =========================
    # TF BROADCAST
    # =========================
    def broadcast_tfs(self, points):

        transforms = []

        for i, pt in enumerate(points):

            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = f"sample_{i}"

            # position
            t.transform.translation.x = pt.x
            t.transform.translation.y = pt.y
            t.transform.translation.z = pt.z

            # default orientation (identity quaternion)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            transforms.append(t)

        # broadcast all at once
        self.br.sendTransform(transforms)

        self.get_logger().info("TFs broadcasted")

        # Optional: keep broadcasting periodically
        self.timer = self.create_timer(0.5, lambda: self.br.sendTransform(transforms))


def main(args=None):
    rclpy.init(args=args)

    node = SampleTFBroadcaster()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()