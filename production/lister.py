import rospy
import tf

rospy.init_node("test_frames")

t = tf.Transformer(True, rospy.Duration(10.0))

print(t.getFrameStrings())

