#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import WrenchStamped
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
        
        # Twist vector 
        omega = np.array([1.0, -0.5, 0.5])
        v = np.array([1.0, -0.5, 1.0])
        self.cV = pin.Motion(v, omega)
        self.twist_pub = self.create_publisher(TwistStamped, 'cage_twist', 10)
        self.get_logger().info('Twist publisher initialized')
        
        # Spatial wrench vector
        force = np.array([0.3, -1.2, 1.0])
        torque = np.array([0.2, -0.3, 0.6])
        self.cW = pin.Force(force, torque)
        self.wrench_pub = self.create_publisher(WrenchStamped, 'cage_wrench', 10)
        self.get_logger().info('Wrench vector initialized')

        # Publish further wrenches 
        self.c2_wrench_pub = self.create_publisher(WrenchStamped, 'c2_wrench', 10)
        self.wrench_world_pub = self.create_publisher(WrenchStamped, 'world_wrench', 10)
        self.world_wrench_pin_pub = self.create_publisher(WrenchStamped, 'world_wrench_pin', 10)
        self.c2_wrench_pin_pub = self.create_publisher(WrenchStamped, 'c2_wrench_pin', 10)
        # Publish twist vectors in world frame + transform to cage corner 2 
        self.world_twist_pub = self.create_publisher(TwistStamped, 'world_twist', 10)
        self.world_twist_pin_pub = self.create_publisher(TwistStamped, 'world_twist_pin', 10)
        self.c2_twist_pub = self.create_publisher(TwistStamped, 'c2_twist', 10)
        self.c2_twist_pin_pub = self.create_publisher(TwistStamped, 'c2_twist_pin', 10)
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
        R = self.cage_center_transform.rotation
        #self.get_logger().info(f'Determinant of R: {np.linalg.det(R)}')
        #self.get_logger().info(f'R.T @ R: {R.T @ R}')
        #self.get_logger().info(f'Type of cage_center_transform: {type(self.cage_center_transform)}')

      
        
    
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
                    #print(entry)
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

        #print(transforms)
        return transforms 


    
    def publish_transforms(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # Transform cV to the world frame using pinocchio.SE3.action
        cV_motion = pin.Motion(self.cV) 
        world_twist_pin = self.cage_center_transform.act(cV_motion)

        # Transform cW to the world frame using pinocchio.SE3.action
        cW_force = pin.Force(self.cW)
        world_wrench_pin = self.cage_center_transform.act(cW_force)

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
    
        

        # Publish Spatial Twist
        twist_msg = TwistStamped()
        twist_msg.header.stamp = now.to_msg()
        twist_msg.header.frame_id = 'cage_corner_6'
        twist_msg.twist.angular.x = float(self.cV.angular[0])
        twist_msg.twist.angular.y = float(self.cV.angular[1])
        twist_msg.twist.angular.z = float(self.cV.angular[2])
        twist_msg.twist.linear.x = float(self.cV.linear[0])
        twist_msg.twist.linear.y = float(self.cV.linear[1])
        twist_msg.twist.linear.z = float(self.cV.linear[2])
        self.twist_pub.publish(twist_msg)


        # Calculate Transformation of cage_corner 6
        world_transform = self.transforms[6]
        R_6 = world_transform.rotation
        p_6 = world_transform.translation
        # Apply twist coordinate transformation
        world_twist = self.twist_coordinate_transformation(self.cV.vector, R_6, p_6)
        
        # World Twist Stamped
        world_twist_msg = TwistStamped()
        world_twist_msg.header.stamp = now.to_msg()
        world_twist_msg.header.frame_id = 'world'
        world_twist_msg.twist.angular.x = float(world_twist[0]) 
        world_twist_msg.twist.angular.y = float(world_twist[1])
        world_twist_msg.twist.angular.z = float(world_twist[2])
        world_twist_msg.twist.linear.x = float(world_twist[3])
        world_twist_msg.twist.linear.y = float(world_twist[4])
        world_twist_msg.twist.linear.z = float(world_twist[5])
        self.world_twist_pub.publish(world_twist_msg) 

        # Calculate Transformation of cage_corner 2 from world frame
        corner_transform_2 = self.transforms[2]
        R_2 = corner_transform_2.rotation
        p_2 = corner_transform_2.translation   
        p_cage = self.cage_center_transform.translation
        c2_twist = self.twist_coordinate_transformation(self.cV.vector, R_2, p_cage + p_2)

        # Corner 2 Twist
        c2_twist_msg = TwistStamped()
        c2_twist_msg.header.stamp = now.to_msg()
        c2_twist_msg.header.frame_id = 'cage_corner_2'
        c2_twist_msg.twist.angular.x = float(c2_twist[0])
        c2_twist_msg.twist.angular.y = float(c2_twist[1])
        c2_twist_msg.twist.angular.z = float(c2_twist[2])
        c2_twist_msg.twist.linear.x = float(c2_twist[3])
        c2_twist_msg.twist.linear.y = float(c2_twist[4])
        c2_twist_msg.twist.linear.z = float(c2_twist[5])
        self.c2_twist_pub.publish(c2_twist_msg) 

        

           # Publish the transformed twist
        world_twist_pin_msg = TwistStamped()
        world_twist_pin_msg.header.stamp = now.to_msg()
        world_twist_pin_msg.header.frame_id = 'world'
        world_twist_pin_msg.twist.angular.x = float(world_twist_pin.angular[0])
        world_twist_pin_msg.twist.angular.y = float(world_twist_pin.angular[1])
        world_twist_pin_msg.twist.angular.z = float(world_twist_pin.angular[2])
        world_twist_pin_msg.twist.linear.x = float(world_twist_pin.linear[0])
        world_twist_pin_msg.twist.linear.y = float(world_twist_pin.linear[1])
        world_twist_pin_msg.twist.linear.z = float(world_twist_pin.linear[2])
        self.world_twist_pin_pub.publish(world_twist_pin_msg)

        # Transform wV to cage_corner_2 using pinocchio.SE3.action
        world_twist_motion = pin.Motion(world_twist_pin)
        c2_twist_pin = corner_transform_2.act(world_twist_motion)
        #self.get_logger().info(f'C2 Twist (Pinocchio): {c2_twist_pin}')

        # Publish the transformed twist
        c2_twist_pin_msg = TwistStamped()
        c2_twist_pin_msg.header.stamp = now.to_msg()
        c2_twist_pin_msg.header.frame_id = 'cage_corner_2'
        c2_twist_pin_msg.twist.angular.x = float(c2_twist_pin.angular[0])
        c2_twist_pin_msg.twist.angular.y = float(c2_twist_pin.angular[1])
        c2_twist_pin_msg.twist.angular.z = float(c2_twist_pin.angular[2])
        c2_twist_pin_msg.twist.linear.x = float(c2_twist_pin.linear[0])
        c2_twist_pin_msg.twist.linear.y = float(c2_twist_pin.linear[1])
        c2_twist_pin_msg.twist.linear.z = float(c2_twist_pin.linear[2])
        self.c2_twist_pin_pub.publish(c2_twist_pin_msg)
        
        # Publish Spatial Wrench
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = now.to_msg()
        wrench_msg.header.frame_id = 'cage_corner_3'
        wrench_msg.wrench.torque.x = float(self.cW.angular[0])
        wrench_msg.wrench.torque.y = float(self.cW.angular[1])
        wrench_msg.wrench.torque.z = float(self.cW.angular[2])
        wrench_msg.wrench.force.x = float(self.cW.linear[0])
        wrench_msg.wrench.force.y = float(self.cW.linear[1])
        wrench_msg.wrench.force.z = float(self.cW.linear[2])
        self.wrench_pub.publish(wrench_msg)

        corner_transform_3 = self.transforms[3]  # Transform von cage_corner_3 zu cage_center
        cage_to_world_transform = self.cage_center_transform  # Transform von cage_center zu world

        # Verkette die Transformationen
        corner_to_world_transform = cage_to_world_transform * corner_transform_3

        # Extrahiere Rotation und Translation
        R_corner_to_world = corner_to_world_transform.rotation
        p_corner_to_world = corner_to_world_transform.translation

        # Wende die Wrench-Transformation an
        world_wrench = self.wrench_coordinate_transformation(self.cW.vector, R_corner_to_world, p_corner_to_world)

        # Publish the transformed wrench
        world_wrench_msg = WrenchStamped()
        world_wrench_msg.header.stamp = now.to_msg()
        world_wrench_msg.header.frame_id = 'world'
        world_wrench_msg.wrench.torque.x = float(world_wrench[0])
        world_wrench_msg.wrench.torque.y = float(world_wrench[1])
        world_wrench_msg.wrench.torque.z = float(world_wrench[2])
        world_wrench_msg.wrench.force.x = float(world_wrench[3])
        world_wrench_msg.wrench.force.y = float(world_wrench[4])
        world_wrench_msg.wrench.force.z = float(world_wrench[5])
        self.wrench_world_pub.publish(world_wrench_msg)
        # Transform world wrench (wW) to cage_center + cage_corner_2
        corner_transform_2 = self.transforms[2]
        world_to_cage_transform = self.cage_center_transform.inverse()  # Transform von world zu cage_center
        world_to_corner_2_transform = corner_transform_2 * world_to_cage_transform

        # Extrahiere Rotation und Translation
        R_2 = world_to_corner_2_transform.rotation
        p_2 = world_to_corner_2_transform.translation

        # Wende die Wrench-Transformation an
        c2_wrench = self.wrench_coordinate_transformation(self.cW.vector, R_2, p_2)

        # Publish the transformed wrench (c2W)
        c2_wrench_msg = WrenchStamped()
        c2_wrench_msg.header.stamp = now.to_msg()
        c2_wrench_msg.header.frame_id = 'cage_corner_2'
        c2_wrench_msg.wrench.torque.x = float(c2_wrench[0])
        c2_wrench_msg.wrench.torque.y = float(c2_wrench[1])
        c2_wrench_msg.wrench.torque.z = float(c2_wrench[2])
        c2_wrench_msg.wrench.force.x = float(c2_wrench[3])
        c2_wrench_msg.wrench.force.y = float(c2_wrench[4])
        c2_wrench_msg.wrench.force.z = float(c2_wrench[5])
        self.c2_wrench_pub.publish(c2_wrench_msg)

        #self.get_logger().info(f'R_6: {R_6}')
        #self.get_logger().info(f'p_6: {p_6}')
        #self.get_logger().info(f'Skew matrix of p_6: {pin.skew(p_6)}')
        #self.get_logger().info(f'Transformed Twist: {world_twist}')
    def twist_coordinate_transformation(self,V,R,p): 
        '''
        Exercise 2.2 
        Spatial Twist Vector

        V: Spatial Twist Vector (6x1)
        R: Rotation Matrix (3x3)
        p: translation vector (3x1)
        '''
        # build adjoint blocks
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R
        Ad[:3, 3:] = pin.skew(p) @ R
        Ad[3:, 3:] = R
        #self.get_logger().info(f'Adjoint Matrix: {Ad}')
        return Ad @ V

    def wrench_coordinate_transformation(self, W, R, p):
        """
        Transform a spatial wrench vector using the adjoint transformation matrix.

        W: Spatial Wrench Vector (6x1)
        R: Rotation Matrix (3x3)
        p: Translation Vector (3x1)
        """
        # Build adjoint transpose blocks for wrench transformation
        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R.T
        AdT[3:, :3] = pin.skew(p) @ R.T
        AdT[3:, 3:] = R.T
        return AdT @ W

             

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
