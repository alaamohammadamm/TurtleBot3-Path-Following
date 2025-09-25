import cv2
import time
import sys
import rclpy
from movement import MoveTB3Command
from camera import ImageAnalysis

# Constants

LINEAR_VEL = 0.1
ANGULAR_VEL = 0.1
K = 1/3

# Initialize Camera

CAMERA_INDEX = 0  # Typically 0 for the first connected camera

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    print("Try changing CAMERA_INDEX (e.g., 0, 1, 2) or ensure camera is connected.")
    sys.exit(1)  # Exit if camera fails to open

print("\nCamera opened successfully. Starting frame capture loop. Press Ctrl+C to quit.")

start_time = time.time()
frame_count = 0
prev_frame = 0

# Initialize Image Analysis

Img_Analyzer = ImageAnalysis()

# Initialize TurtleBot

rclpy.init(args=None)
Bot = MoveTB3Command()

Bot.stop()
time.sleep(5)
print("Sleeping for 5 seconds.")


while rclpy.ok():

    frame_count += 1
    print(f"\n--- Capturing frame {frame_count} ---")
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame from camera. Exiting loop.")
        break

    print(f"Frame {frame_count} successfully read.")

    # Convert BGR (OpenCV default) to RGB if your model expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = Img_Analyzer.follow_line(rgb_frame)

    # TEMPORARY CODE: IMAGE ANALYSIS CODE RETURNS A ANGULAR VELOCITY

    angular_velocity = K * result * ANGULAR_VEL

    if angular_velocity < 0:
        Bot.get_logger().info("Turning Left")
    else:
        Bot.get_logger().info("Turning Right")

    Bot.move(LINEAR_VEL, angular_velocity)


cap.release()
cv2.destroyAllWindows()  # Ensures all OpenCV windows are closed, though not directly used for display here
Bot.stop()
Bot.destroy_node()
rclpy.shutdown()
print("Script finished.")

