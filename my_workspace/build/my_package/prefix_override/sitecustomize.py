import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/djamal/ros2/workspaces/my_workspace/install/my_package'
