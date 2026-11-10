# One detected block, position in the robot base frame (meters, radians).
# x forward from the base axis, y to the left, yaw = long-axis angle in (-pi/2, pi/2].
string color        # "orange" | "green" | "beige"
float32 x
float32 y
float32 yaw
float32 u           # pixel position in the (undistorted) image, for debugging
float32 v
float32 area        # contour area in pixels
