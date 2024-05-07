import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"Camera {index} is working!")
        else:
            print(f"Camera {index} failed to capture a frame.")
    else:
        print(f"Error: Camera {index} could not be opened.")

# Test a range of indices
for i in range(32):  # Adjust this range if necessary
    test_camera(i)
