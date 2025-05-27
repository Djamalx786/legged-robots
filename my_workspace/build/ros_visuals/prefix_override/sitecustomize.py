import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspaces/workspaces/my_workspace/install/ros_visuals'
