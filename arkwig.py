import cv2
import numpy as np
import pigpio
import time

# Setup PiGPIO
pi = pigpio.pi()

# GPIO pins for the servos
servo_left = 12
servo_right = 13

# Initialize servos to neutral position
pi.set_servo_pulsewidth(servo_left, 1500)  # Neutral (typically 1500 for middle)
pi.set_servo_pulsewidth(servo_right, 1500)

def set_servo_angle(pin, angle):
    pulsewidth = int(angle * 1000 / 180 + 1000)
    pi.set_servo_pulsewidth(pin, pulsewidth)

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 15 or idx == 12:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                center_x = (startX + endX) // 2
                
                if center_x < w // 3:
                    set_servo_angle(servo_left, 180)  # Turn left servo to 180 degrees
                    set_servo_angle(servo_right, 0)  # Turn right servo to 0 degrees
                elif center_x > 2 * w // 3:
                    set_servo_angle(servo_left, 0)  # Turn left servo to 0 degrees
                    set_servo_angle(servo_right, 180)  # Turn right servo to 180 degrees
                else:
                    set_servo_angle(servo_left, 90)  # Neutral
                    set_servo_angle(servo_right, 90)  # Neutral

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pi.stop()
