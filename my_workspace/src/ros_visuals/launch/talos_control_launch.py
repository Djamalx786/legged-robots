import launch
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from launch import LaunchDescription
from pathlib import Path


def generate_launch_description():
    # Declare the launch description
    ld = LaunchDescription()
    
    
    rviz_config_dir = os.path.join(FindPackageShare('ros_visuals').find('ros_visuals'), 'rviz')
    rviz_config_file = os.path.join(rviz_config_dir, 'talos.rviz')



    
    talos_description_pkg_share = FindPackageShare('talos_description').find('talos_description')
    urdf_file_path = os.path.join(talos_description_pkg_share, 'robots', 'talos_reduced.urdf')
    #urdf_file_path = "/workspaces/ros2_ws/my_workspace/src/talos_description/robots/talos_reduced.urdf"
    
    with open(urdf_file_path, 'r') as infp:
        robot_desc = infp.read()

    params = {'robot_description': robot_desc}


    # Add the RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file], 
    )
    
    rsp = launch_ros.actions.Node(package='robot_state_publisher',
                                  executable='robot_state_publisher',
                                  output='both',
                                  parameters=[params])
    
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
            "robot_description": robot_desc,
            "use_sim_time": False, 
        }
    ],
    )
    
    t3_node = Node(
        package='bullet_sims',
        executable='t3_main',
        name='t3_main',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )
    """
    t23_node = Node(
        package='bullet_sims',
        executable='t23',
        name='t23',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )
    """
    ld.add_action(rsp)
    #ld.add_action(robot_state_publisher_node)
    ld.add_action(rviz_node)
    ld.add_action(t3_node)
    #ld.add_action(t23_node)
    

    # Return the launch description
    return ld