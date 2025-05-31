# Import necessary modules
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Function to generate the launch description
def generate_launch_description():
    # Initialize the launch description
    launch_description = LaunchDescription()

    # Paths and configurations
    # Group RViz configuration paths
    rviz_config_directory = os.path.join(
        FindPackageShare('ros_visuals').find('ros_visuals'), 'rviz'
    )
    rviz_config_file_path = os.path.join(rviz_config_directory, 'talos.rviz')

    # Group URDF file paths
    talos_description_share = FindPackageShare('talos_description').find('talos_description')
    urdf_file_path = os.path.join(talos_description_share, 'robots', 'talos_reduced.urdf')

    # Read the URDF file
    try:
        with open(urdf_file_path, 'r', encoding='utf-8') as urdf_file:
            robot_description = urdf_file.read()
    except FileNotFoundError:
        print(f"ERROR: The URDF file was not found at {urdf_file_path}")
        print("Please check the path and ensure the 'talos_description' package is built and sourced.")
        robot_description = ""
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading the URDF file: {e}")
        robot_description = ""

    # Nodes
    # Define the RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file_path],
    )

    # Define the Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": False,
            }
        ],
    )

    # Define the RobotSimulation node
    robot_simulation_node = Node(
        package='bullet_sims',
        executable='t23',
        name='TalosSimulator',
        output='screen',
    )

    # Add nodes to the launch description
    launch_description.add_action(robot_state_publisher_node)
    launch_description.add_action(rviz_node)
    launch_description.add_action(robot_simulation_node)

    # Return the launch description
    return launch_description