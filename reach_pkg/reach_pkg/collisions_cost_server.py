#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from custom_interfaces.srv import CollisionCost
from geometry_msgs.msg import Point

import numpy as np


class ProbabilityFieldServer(Node):

    def __init__(self):
        super().__init__('probability_field_server')

        # =========================
        # PARAMETERS
        # =========================
        self.z_seg = 0.7
        self.z_cut = 0.7

        self.width = 0.16

        self.ax_x = 0.15
        self.bx_x = 0.8

        self.ax_y = 0.1
        self.bx_y = 0.4

        self.sigma_z = 1.0

        # Plane patch bounds
        self.x_min = -0.9
        self.x_max = 0.9

        self.y_min = -self.width / 2.0
        self.y_max = self.width / 2.0

        # =========================
        # SERVICE
        # =========================
        self.server = self.create_service(
            CollisionCost,
            'collision_cost',
            self.compute_collisions_callback
        )

        self.get_logger().info("Probability field server ready")

    # ============================================================
    # Density function
    # ============================================================
    def evaluate_density(self, x, y, z):

        # -------------------------
        # Clamp point to rectangle
        # -------------------------
        x_clamped = np.clip(x, self.x_min, self.x_max)
        y_clamped = np.clip(y, self.y_min, self.y_max)

        dx = x - x_clamped
        dy = y - y_clamped
        dz = z - self.z_seg

        # -------------------------
        # Hard cutoff
        # -------------------------
        if z < self.z_cut:
            return 0.0

        # -------------------------
        # Height-dependent spread
        # -------------------------
        z_rel = max(0.0, z - self.z_seg)

        sigma_x = self.ax_x + self.bx_x * z_rel
        sigma_y = self.ax_y + self.bx_y * z_rel

        # -------------------------
        # Density
        # -------------------------
        density = np.exp(
            -(
                dx**2 / sigma_x**2 +
                dy**2 / sigma_y**2 +
                dz**2 / self.sigma_z**2
            )
        )

        return float(density)

    # ============================================================
    # SERVICE CALLBACK
    # ============================================================
    def compute_collisions_callback(self, request, response):

        if len(request.points) == 0:
            response.cost = 0.0
            return response

        probabilities = []

        for p in request.points:

            prob = self.evaluate_density(
                p.x,
                p.y,
                p.z
            )

            probabilities.append(prob)

        # ========================================================
        # OPTION 1: Average cost
        # ========================================================
        cost = np.mean(probabilities)

        # ========================================================
        # OPTION 2: Probabilistic accumulation
        # ========================================================
        # total = 1.0

        # for p in probabilities:
        #     total *= (1.0 - p)

        # cost = 1.0 - total

        response.cost = float(cost)

        self.get_logger().info(
            f"Computed risk from {len(request.points)} points -> {cost:.4f}"
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ProbabilityFieldServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()