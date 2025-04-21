import cv2
import numpy as np
import screen_brightness_control as sbc

roi_top = 100
roi_bottom = 400
roi_left = 250
roi_right = 550

cam = cv2.VideoCapture(0)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand = max(contours, key=cv2.contourArea)

        if cv2.contourArea(hand) > 3000:
            cv2.drawContours(roi, [hand], -1, (0, 255, 0), 2)

            hull = cv2.convexHull(hand, returnPoints=False)
            defects = cv2.convexityDefects(hand, hull)

            if defects is not None:
                fingers = 0

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i][0]
                    start = tuple(hand[s][0])
                    end = tuple(hand[e][0])
                    far = tuple(hand[f][0])

                    a = np.linalg.norm(np.array(start) - np.array(end))
                    b = np.linalg.norm(np.array(start) - np.array(far))
                    c = np.linalg.norm(np.array(end) - np.array(far))

                    angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))

                    if angle < 80:
                        fingers += 1

                total_fingers = fingers + 1
                brightness = min(total_fingers * 20, 100)
                sbc.set_brightness(brightness)

                cv2.putText(frame, f'Brightness: {brightness}%', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 0), 2)
    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
