#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import numpy as np


class PassiveJointPublisher(Node):

    def __init__(self):
        super().__init__('passive_joint_pub')

        # ==========================================================
        # Publisher
        # ==========================================================

        self.pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # ==========================================================
        # Configuration table
        # ==========================================================

        self.table_path = (
            '/home/rosdev/ros2_ws/src/moveit_cpp_demo/'
            'data/final_configuration_table_global2.txt'
        )

        self.load_table()

        # ==========================================================
        # Create the 3D grid
        #
        # Y       -> vineyard
        # Z       -> slider
        # Roll    -> cylinder
        # ==========================================================

        self.y_values = np.sort(
            np.unique(self.table[:, 3])
        )

        self.z_values = np.sort(
            np.unique(self.table[:, 4])
        )

        self.roll_values = np.sort(
            np.unique(self.table[:, 5])
        )

        self.n_y = len(self.y_values)
        self.n_z = len(self.z_values)
        self.n_roll = len(self.roll_values)

        self.get_logger().info(
            f'Table loaded: {len(self.table)} rows'
        )

        self.get_logger().info(
            f'Y:    {self.n_y} values '
            f'[{self.y_values[0]:.6f}, {self.y_values[-1]:.6f}]'
        )

        self.get_logger().info(
            f'Z:    {self.n_z} values '
            f'[{self.z_values[0]:.6f}, {self.z_values[-1]:.6f}]'
        )

        self.get_logger().info(
            f'Roll:  {self.n_roll} values '
            f'[{self.roll_values[0]:.6f}, '
            f'{self.roll_values[-1]:.6f}]'
        )

        # ==========================================================
        # Nested-loop indices
        #
        # Roll changes fastest
        # Z changes next
        # Y changes slowest
        # ==========================================================

        self.i_y = 0
        self.i_z = 0
        self.i_roll = 0

        # ==========================================================
        # One configuration every second
        # ==========================================================

        self.timer = self.create_timer(
            1.0,
            self.publish_next_configuration
        )

        self.get_logger().info(
            'Starting configuration enumeration.'
        )

    # ==============================================================
    # Load configuration table
    # ==============================================================

    def load_table(self):

        try:

            self.table = np.loadtxt(
                self.table_path,
                comments='#'
            )

        except Exception as e:

            self.get_logger().error(
                f'Could not load configuration table:\n{e}'
            )

            raise

        # ----------------------------------------------------------
        # Expected columns:
        #
        # 0  iy
        # 1  iz
        # 2  iroll
        # 3  Y
        # 4  Z
        # 5  Roll
        # 6  candidate
        # 7  quality
        # 8  q0
        # 9  q1
        # 10 q2
        # 11 q3
        # 12 q4
        # 13 q5
        # 14 mode
        # ----------------------------------------------------------

        if self.table.ndim == 1:
            self.table = self.table.reshape(1, -1)

        if self.table.shape[1] < 15:

            raise RuntimeError(
                f'Expected at least 15 columns, '
                f'got {self.table.shape[1]}'
            )

    # ==============================================================
    # Find closest table row
    # ==============================================================

    def find_closest_row(self, y, z, roll):

        # ----------------------------------------------------------
        # Compute squared Euclidean distance in task space
        # ----------------------------------------------------------

        differences = self.table[:, 3:6] - np.array(
            [y, z, roll]
        )

        distances = np.sum(
            differences ** 2,
            axis=1
        )

        index = np.argmin(distances)

        return self.table[index]

    # ==============================================================
    # Publish next configuration
    # ==============================================================

    def publish_next_configuration(self):

        # ----------------------------------------------------------
        # Current desired task-space configuration
        # ----------------------------------------------------------

        y = self.y_values[self.i_y]
        z = self.z_values[self.i_z]
        roll = self.roll_values[self.i_roll]

        # ----------------------------------------------------------
        # Find closest configuration in table
        # ----------------------------------------------------------

        row = self.find_closest_row(
            y,
            z,
            roll
        )

        # ----------------------------------------------------------
        # Extract q0 ... q5
        #
        # Table:
        #
        # Y Z ROLL candidate quality planning q0 q1 q2 q3 q4 q5

        # 0  iy
        # 1  iz
        # 2  iroll
        # 3  Y
        # 4  Z
        # 5  Roll
        # 6  candidate
        # 7  quality
        # 8  q0
        # 9  q1
        # 10 q2
        # 11 q3
        # 12 q4
        # 13 q5
        # 14 mode
        # ----------------------------------------------------------

        #q = row[8:14]
        q = row[9:15] #global

        # ----------------------------------------------------------
        # Publish robot joint state
        # ----------------------------------------------------------

        msg = JointState()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = [
            # Passive joints
            'platform_to_slider',
            'platform_to_cylinder',
            'world_to_vineyard',

            # UR5e joints
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        msg.position = [
            # Passive joints
            z,
            roll,
            y,

            # UR5e joints
            q[0],
            q[1],
            q[2],
            q[3],
            q[4],
            q[5]
        ]

        self.pub.publish(msg)

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------

        table_y = row[3]
        table_z = row[4]
        table_roll = row[5]

        mode = int(row[14])

        self.get_logger().info(
            f'[{self.i_y}, {self.i_z}, {self.i_roll}] '
            f'Y={y:.6f} '
            f'Z={z:.6f} '
            f'Roll={roll:.6f} '
            f'-> '
            f'q=[{q[0]:.3f}, '
            f'{q[1]:.3f}, '
            f'{q[2]:.3f}, '
            f'{q[3]:.3f}, '
            f'{q[4]:.3f}, '
            f'{q[5]:.3f}] '
            f'mode={mode}'
        )

        # Optional: tell us if the closest table point
        # was not exactly the requested point.
        if not (
            np.isclose(y, table_y)
            and np.isclose(z, table_z)
            and np.isclose(roll, table_roll)
        ):

            self.get_logger().warn(
                f'Closest table point differs: '
                f'Y={table_y:.6f}, '
                f'Z={table_z:.6f}, '
                f'Roll={table_roll:.6f}'
            )

        # ----------------------------------------------------------
        # Advance nested loop
        #
        # Roll -> fastest
        # Z    -> next
        # Y    -> slowest
        # ----------------------------------------------------------

        self.i_roll += 1

        if self.i_roll >= self.n_roll:

            self.i_roll = 0
            self.i_z += 1

            if self.i_z >= self.n_z:

                self.i_z = 0
                self.i_y += 1

                if self.i_y >= self.n_y:

                    self.get_logger().info(
                        'Finished all configurations.'
                    )

                    self.timer.cancel()


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