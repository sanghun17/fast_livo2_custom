#!/usr/bin/env python3
"""Divergence guard: kill fast-livo when its odometry runs away.

When fast-livo diverges (e.g. stalled by a CPU/memory burst), the pose shoots
off to infinity and voxblox happily allocates fresh blocks along the runaway
ray until RAM is exhausted. Past ~max_dist meters the estimate is garbage
anyway, so the cheapest defense is to stop the source: SIGINT-equivalent
shutdown of the mapping node via `rosnode kill`. No new pointcloud poses ->
voxblox stops allocating.

FLIGHT NOTE: killing odom also kills /mavros vision_pose input. At 100m+ of
divergence the vehicle has already crashed or is flying blind, so stopping is
still the right call — but keep `enabled` wired to a launch arg so it can be
disabled deliberately for tests that fake large motions.
"""
import rospy
from nav_msgs.msg import Odometry
import subprocess

class OdomGuard:
    def __init__(self):
        self.max_dist = float(rospy.get_param("~max_dist", 100.0))   # meters from origin
        self.target = rospy.get_param("~kill_node", "/laserMapping")
        self.enabled = bool(rospy.get_param("~enabled", True))
        self.tripped = False
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.cb, queue_size=1)
        rospy.loginfo(f"[odom_guard] armed: |pos| > {self.max_dist:.0f}m -> rosnode kill {self.target}"
                      if self.enabled else "[odom_guard] DISABLED (~enabled=false)")

    def cb(self, msg):
        if self.tripped or not self.enabled:
            return
        p = msg.pose.pose.position
        d2 = p.x * p.x + p.y * p.y + p.z * p.z
        if d2 > self.max_dist ** 2:
            self.tripped = True
            rospy.logfatal(f"[odom_guard] ODOMETRY DIVERGED: |pos|={d2 ** 0.5:.1f}m > {self.max_dist:.0f}m "
                           f"-> killing {self.target} (protects voxblox/RAM)")
            # rosnode kill = clean ROS shutdown of the mapping node.
            subprocess.run(["rosnode", "kill", self.target], timeout=10)
            rospy.signal_shutdown("odom diverged, guard tripped")

if __name__ == "__main__":
    rospy.init_node("odom_guard")
    OdomGuard()
    rospy.spin()
