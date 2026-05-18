from setuptools import find_packages, setup

package_name = 'reach_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosdev',
    maintainer_email='deathfromthesky1998@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pruning_positions_service = reach_pkg.node_pruning_positions:main',
            'sample_positions = reach_pkg.node_broadcast_positions:main',
            'sample_orientations = reach_pkg.node_broadcast_orientations:main',
            'orientation_service = reach_pkg.node_angles_cone:main',
            'sample_poses = reach_pkg.node_broadcast_poses:main',
            'passive_joints_platform_publisher = reach_pkg.passive_joints_platform:main',
            'collisions_cost_server = reach_pkg.collisions_cost_server:main'
        ],
    },
)
