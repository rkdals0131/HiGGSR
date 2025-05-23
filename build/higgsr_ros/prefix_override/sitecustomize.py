import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/user1/ROS2_Workspace/higgsros_ws/src/install/higgsr_ros'
