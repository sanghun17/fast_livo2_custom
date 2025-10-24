#!/usr/bin/env python3
"""
Smart Static TF Bridge: Intelligently connects TF trees using static transforms.

Given a desired transform between two frames (e.g., leaves), this script:
1. Checks if the desired transform conflicts with existing TF
2. Finds the root of the child frame's TF tree
3. Calculates the proper parent → root transform by multiplying intermediate transforms
4. Publishes the static transform

Key insight:
- Parent frame can be anywhere (root, middle, or leaf)
- Child frame must be a root (to avoid conflicts)
- Calculate: T(parent → root) = T(parent → child_desired) × T(child_desired → root)

Usage:
    rosrun fast_livo smart_static_tf_bridge.py \
        _parent:=aft_mapped \
        _child:=camera_color_optical_frame \
        _x:=0 _y:=0 _z:=0 \
        _qx:=0 _qy:=0 _qz:=0 _qw:=1

Example (identity transform):
    rosrun fast_livo smart_static_tf_bridge.py \
        _parent:=aft_mapped \
        _child:=camera_color_optical_frame
"""

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import numpy as np
from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_inverse
)


class SmartStaticTFBridge:
    def __init__(self):
        rospy.init_node('smart_static_tf_bridge', anonymous=False)

        # Get desired transform parameters (default: identity)
        self.parent_frame = rospy.get_param('~parent', 'aft_mapped')
        self.child_frame = rospy.get_param('~child', 'camera_color_optical_frame')
        self.x = rospy.get_param('~x', 0.0)
        self.y = rospy.get_param('~y', 0.0)
        self.z = rospy.get_param('~z', 0.0)
        self.qx = rospy.get_param('~qx', 0.0)
        self.qy = rospy.get_param('~qy', 0.0)
        self.qz = rospy.get_param('~qz', 0.0)
        self.qw = rospy.get_param('~qw', 1.0)

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Static transform broadcaster
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        rospy.loginfo("=" * 60)
        rospy.loginfo("Smart Static TF Bridge")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Desired transform: {self.parent_frame} → {self.child_frame}")
        rospy.loginfo(f"  Translation: [{self.x:.4f}, {self.y:.4f}, {self.z:.4f}]")
        rospy.loginfo(f"  Rotation (quat): [{self.qx:.4f}, {self.qy:.4f}, {self.qz:.4f}, {self.qw:.4f}]")

    def wait_for_frame(self, frame_id, timeout=10.0):
        """Wait for a frame to appear in TF tree."""
        rospy.loginfo(f"Waiting for frame '{frame_id}'...")
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logerr(f"Timeout: Frame '{frame_id}' not found!")
                return False

            if frame_id in self.tf_buffer.all_frames_as_string():
                rospy.loginfo(f"  ✓ Frame '{frame_id}' found")
                return True

            rate.sleep()

        return False

    def find_root(self, frame_id):
        """Find the root frame of a TF tree by traversing parents."""
        current = frame_id
        visited = set()
        path = [current]

        while current not in visited and not rospy.is_shutdown():
            visited.add(current)

            # Try to find parent by checking /tf_static
            try:
                msg = rospy.wait_for_message('/tf_static', TFMessage, timeout=2.0)
                parent_found = None

                for transform in msg.transforms:
                    if transform.child_frame_id == current:
                        parent_found = transform.header.frame_id
                        break

                if parent_found is None:
                    # No parent, this is the root
                    rospy.loginfo(f"  Root of '{frame_id}': '{current}'")
                    if len(path) > 1:
                        rospy.loginfo(f"  Path: {' → '.join(reversed(path))}")
                    return current

                # Continue up the tree
                path.append(parent_found)
                current = parent_found

            except Exception as e:
                # Timeout or error, assume this is the root
                rospy.loginfo(f"  Root of '{frame_id}': '{current}' (no parent found)")
                if len(path) > 1:
                    rospy.loginfo(f"  Path: {' → '.join(reversed(path))}")
                return current

        rospy.logwarn(f"Cycle detected in TF tree at '{current}'")
        return current

    def transform_to_matrix(self, trans):
        """Convert transform to 4x4 homogeneous matrix."""
        if isinstance(trans, TransformStamped):
            t = trans.transform
        else:
            t = trans

        q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        mat = quaternion_matrix(q)
        mat[0, 3] = t.translation.x
        mat[1, 3] = t.translation.y
        mat[2, 3] = t.translation.z
        return mat

    def matrix_to_transform(self, matrix, parent, child):
        """Convert 4x4 matrix to TransformStamped."""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent
        t.child_frame_id = child

        t.transform.translation.x = matrix[0, 3]
        t.transform.translation.y = matrix[1, 3]
        t.transform.translation.z = matrix[2, 3]

        q = quaternion_from_matrix(matrix)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        return t

    def lookup_transform_safe(self, parent, child, timeout=1.0):
        """Safely lookup transform with timeout."""
        try:
            trans = self.tf_buffer.lookup_transform(
                parent, child, rospy.Time(0), rospy.Duration(timeout)
            )
            return trans
        except Exception as e:
            rospy.logwarn(f"Cannot lookup {parent} → {child}: {e}")
            return None

    def get_desired_transform_matrix(self):
        """Get the desired transform as a 4x4 matrix."""
        q = [self.qx, self.qy, self.qz, self.qw]
        mat = quaternion_matrix(q)
        mat[0, 3] = self.x
        mat[1, 3] = self.y
        mat[2, 3] = self.z
        return mat

    def calculate_solution(self):
        """
        Calculate the static transform to publish.

        Logic:
        - Desired: T(parent → child)
        - Find root of child frame
        - Calculate: T(parent → root) = T(parent → child) × T(child → root)
        - Publish: parent → root (static)
        """

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("Step 1: Find root of child frame")
        rospy.loginfo("=" * 60)

        child_root = self.find_root(self.child_frame)

        if child_root == self.child_frame:
            rospy.loginfo(f"✓ Child frame '{self.child_frame}' is already a root!")
            rospy.loginfo(f"  Solution: Publish {self.parent_frame} → {self.child_frame} directly")

            # Create transform directly
            T_desired = self.get_desired_transform_matrix()
            solution = self.matrix_to_transform(T_desired, self.parent_frame, self.child_frame)
            return solution

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("Step 2: Calculate transform to root")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Child '{self.child_frame}' has root '{child_root}'")
        rospy.loginfo(f"Need to calculate: {self.parent_frame} → {child_root}")

        # Get T(child → root) = inverse of T(root → child)
        trans_root_to_child = self.lookup_transform_safe(child_root, self.child_frame)

        if trans_root_to_child is None:
            rospy.logerr(f"Failed to lookup {child_root} → {self.child_frame}")
            return None

        T_root_to_child = self.transform_to_matrix(trans_root_to_child)
        T_child_to_root = np.linalg.inv(T_root_to_child)

        rospy.loginfo(f"  Existing: {child_root} → {self.child_frame}")
        t = trans_root_to_child.transform
        rospy.loginfo(f"    t=[{t.translation.x:.4f}, {t.translation.y:.4f}, {t.translation.z:.4f}]")
        rospy.loginfo(f"    q=[{t.rotation.x:.4f}, {t.rotation.y:.4f}, {t.rotation.z:.4f}, {t.rotation.w:.4f}]")

        # Calculate: T(parent → root) = T(parent → child) × T(child → root)
        T_parent_to_child = self.get_desired_transform_matrix()
        T_parent_to_root = T_parent_to_child @ T_child_to_root

        rospy.loginfo(f"  Calculated: {self.parent_frame} → {child_root}")
        solution = self.matrix_to_transform(T_parent_to_root, self.parent_frame, child_root)
        t = solution.transform
        rospy.loginfo(f"    t=[{t.translation.x:.4f}, {t.translation.y:.4f}, {t.translation.z:.4f}]")
        rospy.loginfo(f"    q=[{t.rotation.x:.4f}, {t.rotation.y:.4f}, {t.rotation.z:.4f}, {t.rotation.w:.4f}]")

        return solution

    def run(self):
        """Main execution."""

        # Wait for TF tree to populate
        rospy.loginfo("\nWaiting for TF tree to populate...")
        rospy.sleep(5.0)  # Give TF buffer time to cache all transforms

        # Wait for child frame (parent frame might not exist yet)
        if not self.wait_for_frame(self.child_frame, timeout=10.0):
            rospy.logerr("Cannot proceed without child frame. Exiting.")
            return

        # Calculate solution
        solution = self.calculate_solution()

        if solution is None:
            rospy.logerr("Failed to calculate solution. Exiting.")
            return

        # Publish static transform
        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("Step 3: Publishing static transform")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Publishing: {solution.header.frame_id} → {solution.child_frame_id}")
        t = solution.transform
        rospy.loginfo(f"  Translation: [{t.translation.x:.4f}, {t.translation.y:.4f}, {t.translation.z:.4f}]")
        rospy.loginfo(f"  Rotation: [{t.rotation.x:.4f}, {t.rotation.y:.4f}, {t.rotation.z:.4f}, {t.rotation.w:.4f}]")

        self.static_broadcaster.sendTransform(solution)

        rospy.loginfo("\n✓ Static transform published successfully!")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Static transform is latched on /tf_static topic.")
        rospy.loginfo("Node will stay alive to keep the transform available.")
        rospy.loginfo("=" * 60)

        # Keep node alive to maintain the static transform
        rospy.spin()


if __name__ == '__main__':
    try:
        bridge = SmartStaticTFBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
