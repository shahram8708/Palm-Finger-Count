import cv2
import mediapipe as mp

def detect_finger_status(finger_landmarks):
    finger_tip_y = finger_landmarks.y
    return finger_tip_y

def detect_palm():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    open_count = 0
    closed_count = 0

    prev_finger_open = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        any_palm_detected = False

        if results.multi_hand_landmarks:
            any_palm_detected = True

            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                finger_landmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_tip_y = finger_landmarks.y
                finger_open = finger_tip_y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

                if finger_open != prev_finger_open:
                    if finger_open:
                        print("Finger is open.")
                        open_count += 1
                    else:
                        print("Finger is closed.")
                        closed_count += 1
                    prev_finger_open = finger_open

        cv2.imshow('Palm Detection', frame)

        key = cv2.waitKey(1)
        if key == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

    return open_count, closed_count

def main():
    open_count, closed_count = detect_palm()
    print("Number of second fingers were opened:", open_count)
    print("Number of second fingers were closed:", closed_count)

if __name__ == "__main__":
    main()
