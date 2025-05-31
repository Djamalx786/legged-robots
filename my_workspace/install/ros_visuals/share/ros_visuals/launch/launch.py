from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os



def generate_launch_description():
    # Path to the RViz config file
    rviz_config = os.path.join(
        get_package_share_directory('ros_visuals'), 
        'config',
        'rviz_config.rviz'
    )

    return LaunchDescription([
        # Launch RViz with the saved config
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
        
        # Launch your cage node
        Node(
            package='ros_visuals',
            executable='t13', # choose which exercise (t11, t12 or t13) to run
            name='cage_node',
            output='screen'
        )
    ])