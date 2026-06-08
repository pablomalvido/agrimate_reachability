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

    br1 = Node(
        package='reach_pkg',
        executable='generic_br',
        name='prune_pose_broadcaster',
        output='screen',
        parameters=[{
            'n': 50,
            'tf_prefix': 'pruning_pose_',
            'parent_frame': 'vineyard_base',
            'points_service': 'sample_points',
            'orientations_service': 'sample_orientations'
        }]
    )

    br2 = Node(
        package='reach_pkg',
        executable='generic_br',
        name='scan_pose_broadcaster',
        output='screen',
        parameters=[{
            'n': 30,
            'tf_prefix': 'scaning_pose_',
            'parent_frame': 'vineyard_base',
            'points_service': 'sample_points_scan',
            'orientations_service': 'sample_orientations_scan'
        }]
    )

    br3 = Node(
        package='reach_pkg',
        executable='generic_br',
        name='grasp_pose_broadcaster',
        output='screen',
        parameters=[{
            'n': 30,
            'tf_prefix': 'grasping_pose_',
            'parent_frame': 'vineyard_base',
            'points_service': 'sample_points_grasp',
            'orientations_service': 'sample_orientations_grasp'
        }]
    )

    return LaunchDescription([
        pruning_positions_service,
        orientation_service,
        br1,
        br2,
        br3
    ])