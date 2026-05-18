#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Point
from custom_interfaces.srv import SamplePoints


class SegmentSamplerNode(Node):

    def __init__(self):
        super().__init__('segment_sampler_node')

        # Create service
        self.srv = self.create_service(
            SamplePoints,
            'sample_points',
            self.sample_callback
        )

        self.get_logger().info("Segment sampler service ready")

        # =========================
        # PARAMETERS (same as your script)
        # =========================
        self.A = np.array([-0.9, 0.0, 0.75])
        self.B = np.array([0.9, 0.0, 0.75])
        self.z_seg = self.A[2]

        # spreads
        self.ax_x, self.bx_x = 0.2, 0.0
        self.ax_y, self.bx_y = 0.2, 0.0

        self.az_pos, self.bz_pos = 0.2, 0.0
        self.az_neg, self.bz_neg = 0.15, 0.0

        # bounds
        self.x_min, self.x_max = -1.1, 1.1
        self.y_min, self.y_max = -0.25, 0.25
        self.z_min, self.z_max = 0.7, 1.4

        # precompute
        self.AB = self.B - self.A
        self.AB2 = np.dot(self.AB, self.AB)

    # =========================
    # PROBABILITY FUNCTION
    # =========================
    def compute_density(self, x, y, z):

        P = np.array([x, y, z]) - self.A

        t = np.dot(P, self.AB) / self.AB2
        t = np.clip(t, 0, 1)

        closest = self.A + t * self.AB

        diff = np.array([x, y, z]) - closest
        dx, dy, dz = diff

        z_rel = z - closest[2]

        sigma_x = self.ax_x * (1 + self.bx_x * abs(z_rel))
        sigma_y = self.ax_y * (1 + self.bx_y * abs(z_rel))

        if z_rel >= 0:
            sigma_z = self.az_pos * (1 + self.bz_pos * z_rel)
        else:
            sigma_z = self.az_neg * (1 + self.bz_neg * abs(z_rel))

        density = np.exp(
            -(dx**2 / sigma_x**2 +
              dy**2 / sigma_y**2 +
              dz**2 / sigma_z**2)
        )

        return density

    # =========================
    # SAMPLER
    # =========================
    def sample_points(self, n):

        samples = []

        while len(samples) < n:

            # uniform proposal inside bounds
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            z = np.random.uniform(self.z_min, self.z_max)

            p = self.compute_density(x, y, z)

            # rejection sampling
            if np.random.rand() < p:
                samples.append((x, y, z))

        return samples

    # =========================
    # SERVICE CALLBACK
    # =========================
    def sample_callback(self, request, response):

        n = request.n
        self.get_logger().info(f"Sampling {n} points")

        pts = self.sample_points(n)

        response.points = []
        for x, y, z in pts:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = float(z)
            response.points.append(pt)

        return response


def main(args=None):
    rclpy.init(args=args)

    node = SegmentSamplerNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()