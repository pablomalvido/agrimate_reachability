#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from custom_interfaces.srv import SamplePoints, SampleOrientations

from scipy.spatial.transform import Rotation as R

import tf2_ros


class SamplePoseTFBroadcaster(Node):

    def __init__(self):
        super().__init__('sample_pose_tf_broadcaster')

        # =====================================================
        # Parameters
        # =====================================================

        self.declare_parameter('n', 50)
        self.declare_parameter('tf_prefix', 'sample_pose')
        self.declare_parameter('parent_frame', 'vineyard_base')

        self.declare_parameter(
            'points_service',
            'sample_points'
        )

        self.declare_parameter(
            'orientations_service',
            'sample_orientations'
        )

        self.n = self.get_parameter('n').value
        self.tf_prefix = self.get_parameter('tf_prefix').value
        self.parent_frame = self.get_parameter('parent_frame').value

        points_service = self.get_parameter(
            'points_service'
        ).value

        orientations_service = self.get_parameter(
            'orientations_service'
        ).value

        self.get_logger().info(
            f'n={self.n}, '
            f'tf_prefix={self.tf_prefix}, '
            f'parent_frame={self.parent_frame}'
        )

        # =====================================================
        # TF Broadcaster
        # =====================================================

        self.br = tf2_ros.TransformBroadcaster(self)

        # =====================================================
        # Service Clients
        # =====================================================

        self.points_cli = self.create_client(
            SamplePoints,
            points_service
        )

        self.orient_cli = self.create_client(
            SampleOrientations,
            orientations_service
        )

        while not self.points_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'Waiting for service {points_service}...'
            )

        while not self.orient_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'Waiting for service {orientations_service}...'
            )

        self.get_logger().info('Both services available')

        # =====================================================
        # Data storage
        # =====================================================

        self.points = None
        self.orientations = None
        self.transforms = []

        # Optional rotation offset
        self.rot_offset = R.from_euler(
            'x',
            90,
            degrees=True
        )

        self.request_samples()

    # =========================================================
    # Request samples
    # =========================================================

    def request_samples(self):

        req_p = SamplePoints.Request()
        req_p.n = self.n

        future_p = self.points_cli.call_async(req_p)
        future_p.add_done_callback(
            self.points_callback
        )

        req_o = SampleOrientations.Request()
        req_o.n = self.n

        req_o.max_angle_x = 0.87
        req_o.max_angle_y = 0.87
        req_o.max_yaw = 0.87

        future_o = self.orient_cli.call_async(req_o)
        future_o.add_done_callback(
            self.orient_callback
        )

    # =========================================================
    # Callbacks
    # =========================================================

    def points_callback(self, future):

        try:
            response = future.result()

            self.points = response.points

            self.get_logger().info(
                f"Received {len(self.points)} points"
            )

            self.try_build_transforms()

        except Exception as e:
            self.get_logger().error(
                f"Points service failed: {e}"
            )

    def orient_callback(self, future):

        try:
            response = future.result()

            self.orientations = response.orientations

            self.get_logger().info(
                f"Received {len(self.orientations)} orientations"
            )

            self.try_build_transforms()

        except Exception as e:
            self.get_logger().error(
                f"Orientation service failed: {e}"
            )

    # =========================================================
    # Build TFs
    # =========================================================

    def try_build_transforms(self):

        if self.points is None:
            return

        if self.orientations is None:
            return

        if len(self.points) != len(self.orientations):

            self.get_logger().error(
                "Mismatch in number of points and orientations!"
            )

            return

        self.transforms = []

        for i, (pt, q) in enumerate(
            zip(self.points, self.orientations)
        ):

            t = TransformStamped()

            t.header.frame_id = self.parent_frame
            t.child_frame_id = (
                f"{self.tf_prefix}_{i}"
            )

            # Position
            t.transform.translation.x = pt.x
            t.transform.translation.y = pt.y
            t.transform.translation.z = pt.z

            # Orientation
            q_orig = [q.x, q.y, q.z, q.w]

            rot = R.from_quat(q_orig)

            rot_new = self.rot_offset * rot

            q_new = rot_new.as_quat()

            t.transform.rotation.x = q_new[0]
            t.transform.rotation.y = q_new[1]
            t.transform.rotation.z = q_new[2]
            t.transform.rotation.w = q_new[3]

            self.transforms.append(t)

        self.get_logger().info(
            f"Built {len(self.transforms)} TFs"
        )

        self.timer = self.create_timer(
            0.1,
            self.broadcast_tfs
        )

    # =========================================================
    # Broadcast
    # =========================================================

    def broadcast_tfs(self):

        if not self.transforms:
            return

        now = self.get_clock().now().to_msg()

        for t in self.transforms:
            t.header.stamp = now

        self.br.sendTransform(
            self.transforms
        )


def main(args=None):

    rclpy.init(args=args)

    node = SamplePoseTFBroadcaster()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()