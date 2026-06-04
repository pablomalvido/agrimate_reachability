from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    pruning_positions_service = Node(
        package='reach_pkg',
        executable='pruning_positions_service',
        name='pruning_positions_service',
        output='screen'
    )

    orientation_service = Node(
        package='reach_pkg',
        executable='orientation_service',
        name='orientation_service',
        output='screen'
    )

    sample_poses = Node(
        package='reach_pkg',
        executable='sample_poses',
        name='sample_poses',
        output='screen'
    )

    return LaunchDescription([
        pruning_positions_service,
        orientation_service,
        sample_poses
    ])