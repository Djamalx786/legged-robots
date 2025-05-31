#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from visualization_msgs.msg import Marker

class CageNode(Node):
    def __init__(self):
        super().__init__('cage_node')
        self.get_logger().info('Node started!')

        # Parameters
        self.declare_parameter('size', 2.0)
        self.declare_parameter('method', 'exp6')
        self.cage_size = self.get_parameter('size').value
        self.method = self.get_parameter('method').value

        # Motion parameters
        self.linear_vel_world = np.array([0.5, 0.0, 0.0])
        self.angular_vel_world = np.array([0.0, 0.0, 1.0])
        self.linear_vel_body = np.array([0.5, 0.0, 0.0])
        self.angular_vel_body = np.array([0.0, 0.0, 1.0])

        # Point and publishers
        self.p = np.array([0, 0.5, 0.5])
        self.corner_marker_pub = self.create_publisher(Marker, 'corner_point', 10)
        self.world_marker_pub = self.create_publisher(Marker, 'world_point', 10)

        # TF Broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.last_time = self.get_clock().now()

        # Initialize cage_center transform
        self.cage_center_transform = pin.SE3.Identity()
        self.cage_center_transform.translation = np.array([0.5, 0.0, 0.5])
        self.cage_center_transform.rotation = (
            self.axis_angle_to_rotation_matrix(np.array([0,0,1]), np.pi/4)
            @ self.axis_angle_to_rotation_matrix(np.array([1,0,0]), np.pi/6)
        )

        # Publish static corner offsets once
        self.transforms = self.generate_cage_transforms()
        static_msgs = []
        now_msg = self.get_clock().now().to_msg()
        for i, tf in enumerate(self.transforms):
            msg = TransformStamped()
            msg.header.stamp = now_msg
            msg.header.frame_id = 'cage_center'
            msg.child_frame_id = f'cage_corner_{i}'
            msg.transform.translation.x = float(tf.translation[0])
            msg.transform.translation.y = float(tf.translation[1])
            msg.transform.translation.z = float(tf.translation[2])
            q = pin.Quaternion(tf.rotation)
            msg.transform.rotation.x = q.x
            msg.transform.rotation.y = q.y
            msg.transform.rotation.z = q.z
            msg.transform.rotation.w = q.w
            static_msgs.append(msg)
        self.static_tf_broadcaster.sendTransform(static_msgs)

        # Timer
        self.timer = self.create_timer(0.1, self.publish_transforms)

    def axis_angle_to_rotation_matrix(self, axis, theta):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta/2.0)
        b, c, d = -axis * np.sin(theta/2.0)
        return np.array([
            [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
            [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
        ])

    def normalize(v):
        return v / np.linalg.norm(v)

    def generate_cage_transforms(self):
        rotation = [
        [0, 1],                     # Corner 0: no rotation around X
        [np.pi/2, 2],               # Corner 1: +90° around Y
        [-np.pi/2, 3],               # Corner 2: -90° around Z
        [-np.pi, 1],                # Corner 3: 180° around X
        [np.pi/2, 3],              # Corner 4: +90° around Z
        [-np.pi, 2],               # Corner 5: +90° around Z
        [np.pi, 3],                 # Corner 6: 180° around Z
        [-np.pi, 1, np.pi/2,3]    # Corner 7: +90° around Y, then +90° around Z
    ]
        half = self.cage_size / 2.0
        transforms = []
        i = 0
        for x in [-half, half]:
            for y in [-half, half]:
                for z in [-half, half]:
                    entry = rotation[i]
                    i += 1
                    print(entry)
                    # Starte mit Einheitsmatrix
                    R = np.eye(3)

                    # Wende alle Rotationen der Reihe nach an
                    for j in range(0, len(entry), 2):
                        angle = entry[j]
                        axis = entry[j + 1]

                        if axis == 1:
                            axis_vector = np.array([1.0, 0.0, 0.0])
                        elif axis == 2:
                            axis_vector = np.array([0.0, 1.0, 0.0])
                        elif axis == 3:
                            axis_vector = np.array([0.0, 0.0, 1.0])
                        
                        R_step = pin.AngleAxis(angle, axis_vector).toRotationMatrix()
                        R = R @ R_step  # Hintereinanderausführung
                    M = pin.SE3(R, np.array([x,y,z]))
                    transforms.append(M)

        print(transforms)
        return transforms 


    
    def publish_transforms(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # Update cage_center pose
        if self.method == 'lecture':
            self.cage_center_transform.translation += self.linear_vel_world * dt
            rot_inc = self.axis_angle_to_rotation_matrix(
                self.angular_vel_world, np.linalg.norm(self.angular_vel_world)*dt)
            self.cage_center_transform.rotation = rot_inc @ self.cage_center_transform.rotation
        else:
            twist = pin.Motion(
                self.linear_vel_body * dt, self.angular_vel_body * dt)
            self.cage_center_transform = self.cage_center_transform * pin.exp6(twist)

        # Publish dynamic center transform with world as parent
        msg_center = TransformStamped()
        msg_center.header.stamp = now.to_msg()
        msg_center.header.frame_id = 'world'
        msg_center.child_frame_id = 'cage_center'
        msg_center.transform.translation.x = float(self.cage_center_transform.translation[0])
        msg_center.transform.translation.y = float(self.cage_center_transform.translation[1])
        msg_center.transform.translation.z = float(self.cage_center_transform.translation[2])
        q = pin.Quaternion(self.cage_center_transform.rotation)
        msg_center.transform.rotation.x = q.x
        msg_center.transform.rotation.y = q.y
        msg_center.transform.rotation.z = q.z
        msg_center.transform.rotation.w = q.w
        self.tf_broadcaster.sendTransform(msg_center)

        # Publish corner marker in cage_center frame
        corner_marker = Marker()
        corner_marker.header.frame_id = 'cage_center'
        corner_marker.header.stamp = now.to_msg()
        corner_marker.ns = 'corner_point'
        corner_marker.id = 0
        corner_marker.type = Marker.SPHERE
        corner_marker.action = Marker.ADD
        corner_marker.scale.x = corner_marker.scale.y = corner_marker.scale.z = 0.23
        corner_marker.color.r = 1.0
        corner_marker.color.a = 1.0
        # marker fixed relative
        corner_marker.pose.position.x = float(self.p[0])
        corner_marker.pose.position.y = float(self.p[1])
        corner_marker.pose.position.z = float(self.p[2])
        corner_marker.frame_locked = True
        self.corner_marker_pub.publish(corner_marker)

        # Publish world marker in world frame
        world_point = self.cage_center_transform.act(pin.SE3(np.eye(3), self.p))
        world_marker = Marker()
        world_marker.header.frame_id = 'world'
        world_marker.header.stamp = now.to_msg()
        world_marker.ns = 'world_point'
        world_marker.id = 0
        world_marker.type = Marker.SPHERE
        world_marker.action = Marker.ADD
        world_marker.scale.x = world_marker.scale.y = world_marker.scale.z = 0.15
        world_marker.color.b = 1.0
        world_marker.color.a = 1.0
        world_marker.pose.position.x = float(world_point.translation[0])
        world_marker.pose.position.y = float(world_point.translation[1])
        world_marker.pose.position.z = float(world_point.translation[2])
        world_marker.frame_locked = True
        self.world_marker_pub.publish(world_marker)


def main(args=None):
    rclpy.init(args=args)
    node = CageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
