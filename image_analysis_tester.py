import cv2
import time
from picamera2 import Picamera2
from camera import ImageAnalysis

def main():
    try:
        cam = Picamera2()
        #cam.framerate = 100
        cam.configure(cam.create_preview_configuration(main={"format": "BGR888"}))
        cam.start()
        camera = ImageAnalysis()

        while True:
            #print("program is running...")
            #time.sleep(2)
            frame = cam.capture_array()
            #cv2.imshow("Camera Feed", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(50)
            result = camera.follow_line(frame)
            #cv2.imwrite("test_image.jpg", result)
            print(result)

    except Exception as e:
        print("***ERROR OCCURRED***")
        print(e)
        exit()

    finally:
        print("closing")
        cap.release()

main()
