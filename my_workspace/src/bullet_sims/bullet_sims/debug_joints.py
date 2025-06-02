#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class JointDebugger(Node):
    def __init__(self):
        super().__init__('joint_debugger')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        self.received_count = 0

    def joint_state_callback(self, msg):
        self.received_count += 1
        if self.received_count <= 3:  # Show first 3 messages
            print(f"\n=== MESSAGE {self.received_count} ===")
            print("Joint names and positions:")
            for i, (name, pos) in enumerate(zip(msg.name, msg.position)):
                if abs(pos) > 0.001:  # Only show joints that are moving
                    print(f"Index {i:2d}: {name:25s} = {pos:8.4f}")

def main():
    rclpy.init()
    debugger = JointDebugger()
    
    print("Listening to /joint_states...")
    print("Move ONE joint in your simulation and watch the output!")
    
    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        pass
    finally:
        debugger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()