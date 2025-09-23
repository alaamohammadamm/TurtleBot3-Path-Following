import cv2
import numpy as np
import time

class ImageAnalysis:
    # MUST be False for TurtleBots
    DISPLAY_IMG = False
    RGB_LOW = (0,0,150)
    RGB_HIGH = (125,125,255)

    def __init__(self):
        print("initializing camera...")

    def follow_line(self, img):
        self.H, self.W, _, = img.shape
        masked_img = self.mask_image(img)
        cx = self.find_centroid(masked_img)
        self.movement_correction(cx)
        return self.correction

    def mask_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = cv2.inRange(img, self.RGB_LOW, self.RGB_HIGH)
        line_img = gray & mask
        if self.DISPLAY_IMG:
             cv2.imshow("masked image", line_img)
             cv2.waitKey(1)
        return line_img

    def find_centroid(self, masked_img):
        contours, _ = cv2.findContours(np.uint8(masked_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            line = max(contours, key=cv2.contourArea)
            if cv2.contourArea(line) > 3000:
                moments = cv2.moments(line)
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])

                if self.DISPLAY_IMG:
                    cv2.drawContours(masked_img, contours, -1, (180, 255, 180), 3)
                    cv2.drawContours(masked_img, line, -1, (0, 255, 0), 3)
                    cv2.circle(masked_img, (cx, cy), 4, (255, 0, 0), -1)
                    cv2.imshow("centroid", masked_img)
                    cv2.waitKey(1)

            else:
                cx = -1
        else:
            cx = -1
        return cx

    def movement_correction(self, cx):
        midpoint = (self.W)/2
        self.correction = (cx - midpoint)/midpoint
