#!/usr/bin/env python3
"""
Smart Static TF Bridge: Intelligently connects TF trees using static transforms.

One node manages a LIST of independent (parent, child) bridges. For each:
1. Checks if the desired transform conflicts with existing TF
2. Finds the root of the child frame's TF tree
3. Calculates the proper parent → root transform by multiplying intermediate transforms
4. Publishes the static transform

Key insight:
- Parent frame can be anywhere (root, middle, or leaf)
- Child frame must be a root (to avoid conflicts)
- Calculate: T(parent → root) = T(parent → child_desired) × T(child_desired → root)

Usage (one process manages any number of bridges, via launch file):
    <node pkg="fast_livo" type="smart_static_tf_bridge.py" name="smart_tf_bridge" output="log">
        <rosparam param="bridges">
            - {parent: aft_mapped, child: camera_imu_optical_frame}
            - {parent: aft_mapped, child: base_link}
        </rosparam>
    </node>
    (any key omitted per-item defaults to: parent=aft_mapped, child=camera_color_optical_frame,
     translation 0, rotation identity, verify_root=odom)
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


class Bridge:
    """One (parent, child) static-TF spec, its published/done state, and the
    logic to resolve + publish it. Shares a single tf_buffer across all bridges
    in the node (injected, not owned) so the node only opens one /tf listener."""

    def __init__(self, spec, tf_buffer):
        self.tf_buffer = tf_buffer

        # Desired transform parameters (default: identity)
        self.parent_frame = spec.get('parent', 'aft_mapped')
        self.child_frame = spec.get('child', 'camera_color_optical_frame')
        self.x = spec.get('x', 0.0)
        self.y = spec.get('y', 0.0)
        self.z = spec.get('z', 0.0)
        self.qx = spec.get('qx', 0.0)
        self.qy = spec.get('qy', 0.0)
        self.qz = spec.get('qz', 0.0)
        self.qw = spec.get('qw', 1.0)
        self.verify_root = spec.get('verify_root', 'odom')

        self.published = False
        self.done = False

        rospy.loginfo("=" * 60)
        rospy.loginfo("Smart Static TF Bridge")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Desired transform: {self.parent_frame} → {self.child_frame}")
        rospy.loginfo(f"  Translation: [{self.x:.4f}, {self.y:.4f}, {self.z:.4f}]")
        rospy.loginfo(f"  Rotation (quat): [{self.qx:.4f}, {self.qy:.4f}, {self.qz:.4f}, {self.qw:.4f}]")

    def find_root(self, frame_id):
        """Find the root frame of a TF tree by traversing parents.

        Reads parentage from the accumulated tf_buffer (all_frames_as_yaml), which
        holds EVERY latched static + dynamic transform the listener has seen.
        Do NOT use wait_for_message('/tf_static'): that returns ONE message from
        ONE publisher — with several static broadcasters up (camera, mavros, this
        bridge), whichever answers first wins, and if it is e.g. mavros's
        (map_ned/odom_ned), the camera frames are absent from it and the child is
        misdeclared a root -> the step() guard then retries forever ("Camera tree
        not formed yet" loop) even though the tree is fully formed.
        """
        import yaml
        try:
            frames = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
        except Exception as e:
            rospy.logwarn(f"all_frames_as_yaml parse failed: {e}")
            frames = {}

        current = frame_id
        path = [current]
        while not rospy.is_shutdown():
            info = frames.get(current)
            parent = info.get('parent') if isinstance(info, dict) else None
            if not parent:
                rospy.loginfo(f"  Root of '{frame_id}': '{current}'")
                if len(path) > 1:
                    rospy.loginfo(f"  Path: {' → '.join(reversed(path))}")
                return current
            if parent in path:
                rospy.logwarn(f"Cycle detected in TF tree at '{parent}'")
                return current
            path.append(parent)
            current = parent
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

    def step(self, broadcaster):
        """One retry-loop iteration: check child present -> check tree formed ->
        calculate_solution -> publish -> check verify_root->child resolved.

        One-shot publishing races a still-forming camera tree on startup (especially
        under CPU load): the bridge can fire before the RealSense static TFs are
        visible, so find_root() prematurely returns the child itself and we publish a
        direct aft_mapped->child that later COLLIDES with the child's real parent ->
        a split/conflicting tree (the "odom and camera_depth_optical_frame not
        connected" symptom). So instead of a magic sleep, keep recomputing and
        republishing (static sendTransform is idempotent) and only stop once
        verify_root->child looks up end-to-end -- which also proves the live
        camera_init->aft_mapped link from LIO is up, not just our own static.
        """
        # Need the child frame before we can find its root.
        if self.child_frame not in self.tf_buffer.all_frames_as_string():
            rospy.loginfo_throttle(10, f"Waiting for child frame '{self.child_frame}'...")
            return self.done

        # Guard the startup race: if find_root() still thinks the child is its own
        # root, the camera tree's static TFs aren't visible yet. Publishing now
        # would latch a wrong-root transform that conflicts once they appear.
        if self.find_root(self.child_frame) == self.child_frame:
            rospy.loginfo_throttle(10, f"Camera tree for '{self.child_frame}' not formed yet; retrying...")
            return self.done

        solution = self.calculate_solution()
        if solution is None:
            return self.done

        broadcaster.sendTransform(solution)
        if not self.published:
            rospy.loginfo(f"Published {solution.header.frame_id} -> {solution.child_frame_id}; "
                          f"verifying {self.verify_root} -> {self.child_frame} ...")
            self.published = True

        # End-to-end check: confirms aft_mapped is linked all the way to the global
        # root (live LIO odometry), not just to our own static transform.
        if self.lookup_transform_safe(self.verify_root, self.child_frame, timeout=1.0) is not None:
            rospy.loginfo(f"✓ TF tree connected: {self.verify_root} -> {self.child_frame}. Holding transform.")
            self.done = True
            return self.done

        rospy.loginfo_throttle(10, f"Published, but {self.verify_root} -> {self.child_frame} not resolvable yet; retrying...")
        return self.done


def main():
    rospy.init_node('smart_static_tf_bridge', anonymous=False)

    # ~bridges: list of dicts, one per (parent, child) pair. The only supported
    # interface -- every caller sets it (no flat-param single-bridge mode).
    specs = rospy.get_param('~bridges', None)
    if not specs:
        rospy.logfatal("~bridges rosparam missing/empty -- nothing to bridge. Set a list "
                        "of {parent, child, ...} dicts on this node.")
        return

    # Shared buffer/listener/broadcaster: one /tf subscription for every bridge,
    # not N. Single-threaded round-robin below, no threads/locks needed.
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    bridges = [Bridge(spec, tf_buffer) for spec in specs]

    rospy.loginfo("\nWaiting for TF tree to populate...")
    rospy.sleep(2.0)  # brief settle; the retry loop below handles the rest

    rate = rospy.Rate(0.5)  # retry every 2s
    while not rospy.is_shutdown() and not all(b.done for b in bridges):
        for b in bridges:
            if not b.done:
                b.step(broadcaster)
        if not all(b.done for b in bridges):
            rate.sleep()

    # Keep node alive to maintain the latched static transforms.
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
