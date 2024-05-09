import cv2
import mediapipe as mp

def detect_open_fingers(hand_landmarks, mp_hands, handedness):
    finger_open = [False] * 5

    if handedness == 'Right':
        finger_open[0] = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x >
                          hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)
    else:
        finger_open[0] = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x <
                          hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)

    fingertip_indices = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    pip_indices = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    for i, (tip_idx, pip_idx) in enumerate(zip(fingertip_indices, pip_indices)):
        finger_open[i + 1] = (hand_landmarks.landmark[tip_idx].y <
                              hand_landmarks.landmark[pip_idx].y)

    return finger_open

def detect_palm():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    cv2.namedWindow('Palm Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Palm Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_with_annotations = frame.copy()
        total_open_count = 0  

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness = handedness_info.classification[0].label
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_with_annotations, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_statuses = detect_open_fingers(hand_landmarks, mp_hands, handedness)
                total_open_count += sum(finger_statuses) 

        cv2.putText(frame_with_annotations, f'Open Fingers: {total_open_count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Palm Detection', frame_with_annotations)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    detect_palm()

if __name__ == "__main__":
    main()
