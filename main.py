import cv2
import time
import sys  # Import sys for exiting the script
from model import TFLiteModel  # Import your TFLiteModel class
import rclpy
from movement import MoveTB3Command
import os

# --- Configuration ---
MODEL_PATH = 'model_files/model.tflite'
LABELS_PATH = 'model_files/labels.txt'
CAMERA_INDEX = 0  # Typically 0 for the first connected camera

# Initialize the TFLiteModel
try:
    model_handler = TFLiteModel(MODEL_PATH, LABELS_PATH)
except Exception as e:
    print(f"Error initializing TFLiteModel: {e}")
    print("Please check MODEL_PATH and LABELS_PATH configuration.")
    sys.exit(1)  # Exit if model loading fails

# Initialize Camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    print("Try changing CAMERA_INDEX (e.g., 0, 1, 2) or ensure camera is connected.")
    sys.exit(1)  # Exit if camera fails to open

print("\nCamera opened successfully. Starting frame capture loop. Press Ctrl+C to quit.")

rclpy.init(args=None)
Bot = MoveTB3Command()

start_time = time.time()
frame_count = 0
prev_frame = 0

Bot.stop()
time.sleep(5)
print("Sleeping for 5 seconds.")


def process_command(angle):

    if angle == -1:
        Bot.get_logger().info("Turning Sharply Left")
    elif angle == -0.5:
        Bot.get_logger().info("Turning Slightly Left")
    elif angle == 0:
        Bot.get_logger().info("Moving Forward")
    elif angle == 0.5:
        Bot.get_logger().info("Turning Slightly Right")
    elif angle == 1:
        Bot.get_logger().info("Turning Sharply Right")
    else:
        Bot.get_logger().info("NO KNOWN COMMAND")

    Bot.move(Bot.linear_speed, Bot.angular_speed * angle)


while rclpy.ok():

    """

    print(f"\n--- Capturing frame {frame_count} ---")
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame from camera. Exiting loop.")
        break

    print(f"Frame {frame_count} successfully read.")

    # Convert BGR (OpenCV default) to RGB if your model expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform prediction using the model handler
    predicted_label, confidence, inference_time_ms = model_handler.predict(rgb_frame)

    display_text = f"Label: {predicted_label}, Conf: {confidence:.2f}"

    if confidence > 0.7:
        predicted_label = predicted_label[2:]

    if predicted_label == "Stop":
        break

    process_command(predicted_label)

    frame_count += 1
    
    """

    turn_angle = int(input("Give me one of the following numbers: -1, -0.5, 0, 0.5, 1"))  # Value between -1 and 1 that specifies angular speed.
    process_command(turn_angle)




cap.release()
cv2.destroyAllWindows()  # Ensures all OpenCV windows are closed, though not directly used for display here
Bot.stop()
Bot.destroy_node()
rclpy.shutdown()
print("Script finished.")

