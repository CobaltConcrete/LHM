import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# FORCE MJPEG (critical for WSL + usbipd cameras)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Set a conservative resolution first
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    if not ret:
        print("frame failed")
        break

    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()