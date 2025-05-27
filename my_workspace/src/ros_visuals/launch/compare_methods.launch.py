from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('ros_visuals'),
        'config',
        'compare_cage.rviz'
    )

    return LaunchDescription([
        # Lecture-Methode Node
        Node(
            package='ros_visuals',
            executable='t11',
            name='lecture_cage',
            parameters=[{'method': 'lecture', 'tf_prefix': 'lecture'}],
            output='screen'
        ),
        
        # Exp6-Methode Node
        Node(
            package='ros_visuals',
            executable='t11',
            name='exp6_cage',
            parameters=[{'method': 'exp6', 'tf_prefix': 'exp6'}],
            output='screen'
        ),
        
        # RViz mit Konfiguration
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config],
            output='screen'
        )
    ])