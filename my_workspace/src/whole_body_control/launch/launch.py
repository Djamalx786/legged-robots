import launch
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from launch import LaunchDescription


def generate_launch_description():
    # Declare the launch description
    ld = LaunchDescription()
    talos_description_pkg_share = FindPackageShare('talos_description').find('talos_description')
    urdf_file_path = os.path.join(talos_description_pkg_share, 'robots', 'talos_reduced_no_hands.urdf')

    # Declare the RViz configuration file path
    rviz_config_dir = os.path.join(FindPackageShare('ros_visuals').find('ros_visuals'), 'config')
    rviz_config_file = os.path.join(rviz_config_dir, 'talos.rviz')    

    with open(urdf_file_path, 'r') as urdf_temp:
        robot_description = urdf_temp.read()

    params = {'robot_description': robot_description}


    # Add the RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file], 
    )
    
    robot_state_publisher = launch_ros.actions.Node(package='robot_state_publisher',
                                  executable='robot_state_publisher',
                                  output='both',
                                  parameters=[params])
    
    
    t4_node = Node(
        package='whole_body_control',
        executable='t4_standing',
        name='t4_standing_unique',
        output='screen',
        parameters=[{'use_sim_time': False}],
        #environment={'PYTHONPATH': '/workspaces/workspaces/my_workspace/src:/workspaces/workspaces/my_workspace/src/whole_body_control'},
    )
    one_leg_stand_node = Node(
        package='whole_body_control',
        executable='02_one_leg_stand',
        name='one_leg_stand',
        output='screen',
        parameters=[{'use_sim_time': False}],
        #environment={'PYTHONPATH': '/workspaces/workspaces/my_workspace/src:/workspaces/workspaces/my_workspace/src/whole_body_control'},
    )
    squatting_node = Node(
        package='whole_body_control',
        executable='03_squating',
        name='squatting',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )
    # Add the nodes to the launch description
    ld.add_action(robot_state_publisher)
    ld.add_action(rviz_node)
    #ld.add_action(t4_node)
    #ld.add_action(one_leg_stand_node)
    ld.add_action(squatting_node)
    
    return ld