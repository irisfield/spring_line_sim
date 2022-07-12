#!/usr/bin/env python3

import cv2
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
from dynamic_reconfigure.server import Server
from fictitious_line_sim.cfg import ControlUnitConfig

# global variables
vel_msg = Twist()

# since the car starts at the yellow line, drive the first curve
drive_curve = True

################### callback ###################

def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    return config

def yaw_rate_callback(angular_z):
    global yaw_rate
    yaw_rate = angular_z.data
    return

def detect_yellow_callback(yellow_detected):
    global vel_msg, drive_curve

    if RC.enable_drive:
        if yellow_detected.data and not drive_curve:
            # drive straight for x seconds up the yellow line
            # drive_duration(0.5, 0.0, 0.95)
            drive_curve = True
        elif not yellow_detected.data and drive_curve:
            # from the yellow line, drive the curve for x seconds
            # drive_duration(1.0, 0.123, 2.3)
            drive_curve = False
        else:
            # engage the line following algorithm
            vel_msg.linear.x = RC.speed
            vel_msg.angular.z = yaw_rate

            # this message is being published by drive_duration
            cmd_vel_pub.publish(vel_msg)
    else:
        stop_vehicle()

    return

################### helper functions ###################

def drive_duration(speed, yaw_rate, duration):
    time_start = rospy.Time.now()
    time_elapsed = 0.0

    while(time_elapsed <= duration):
        # compute elapsed time in seconds
        time_elapsed = (rospy.Time.now() - time_start).to_sec()

        vel_msg.linear.x = speed
        vel_msg.angular.z = yaw_rate

        cmd_vel_pub.publish(vel_msg)

    # stop the vehicle after driving the duration
    stop_vehicle()

    return

def stop_vehicle():
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    cmd_vel_pub.publish(vel_msg)
    return

################### main ###################

if __name__ == "__main__":
    rospy.init_node("control_unit", anonymous=True)

    rospy.Subscriber("yellow_detected", Bool, detect_yellow_callback)
    rospy.Subscriber("yaw_rate", Float32, yaw_rate_callback)

    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    dynamic_reconfigure_server = Server(ControlUnitConfig, dynamic_reconfigure_callback)

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
