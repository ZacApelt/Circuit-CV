import cv2

# connect to IP camera
cap = cv2.VideoCapture("http://10.89.76.82:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # scale the frame
    frame = cv2.resize(frame, (640, 480))

    # display the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("./circuits/frame.png", frame)
        print("Frame saved")
