import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Quaternion
from custom_interfaces.srv import SampleOrientations

from scipy.spatial.transform import Rotation as R


class ConeOrientationSampler(Node):

    def __init__(self):
        super().__init__('cone_orientation_sampler')

        self.srv = self.create_service(
            SampleOrientations,
            'sample_orientations',
            self.callback
        )

        self.get_logger().info("Cone sampler with roll constraint ready.")

    def sample_orientation(self, max_x, max_y, max_yaw):
        # 1. sample tilt angles (now around X and Y)
        theta_x = np.random.uniform(-max_x, max_x)
        theta_y = np.random.uniform(-max_y, max_y)

        # 2. construct target Z-axis direction
        vec = np.array([
            np.tan(theta_x),
            np.tan(theta_y),
            1.0
        ])
        vec = vec / np.linalg.norm(vec)

        # 3. rotation aligning Z-axis → vec
        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(z_axis, vec)
        c = np.dot(z_axis, vec)

        if np.linalg.norm(v) < 1e-8:
            rot_align = R.identity()
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            rot_matrix = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
            rot_align = R.from_matrix(rot_matrix)

        # 4. constrained yaw around final Z-axis
        yaw = np.random.uniform(-max_yaw, max_yaw)
        rot_yaw = R.from_rotvec(yaw * vec)

        # 5. final rotation
        rot = rot_yaw * rot_align

        return rot.as_quat()

    def callback(self, request, response):
        N = request.n
        max_x = request.max_angle_x
        max_y = request.max_angle_y
        max_yaw = request.max_yaw

        orientations = []

        for _ in range(N):
            q = self.sample_orientation(max_x, max_y, max_yaw)

            msg = Quaternion()
            msg.x = q[0]
            msg.y = q[1]
            msg.z = q[2]
            msg.w = q[3]

            orientations.append(msg)

        response.orientations = orientations

        self.get_logger().info(
            f"Generated {N} samples "
            f"(tilt_x={max_x:.2f}, tilt_y={max_y:.2f}, yaw={max_yaw:.2f})"
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ConeOrientationSampler()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()