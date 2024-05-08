import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)  

people_count = 0
previous_count = 0
no_people_count = 0
threshold_no_people = 50 
aspect_ratio_threshold = 0.8  
total_people_detected = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    rects, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    rects_filtered = []
    for (x, y, w, h) in rects:
        if h / w > aspect_ratio_threshold:  
            rects_filtered.append((x, y, w, h))

    current_count = len(rects_filtered)
    if current_count > previous_count:
        people_count += current_count - previous_count
    elif current_count < previous_count:
        people_count -= previous_count - current_count
    previous_count = current_count

    for (x, y, w, h) in rects_filtered:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Human Detection', frame)

    if len(rects_filtered) == 0:
        no_people_count += 1
    else:
        no_people_count = 0

    if no_people_count >= threshold_no_people:
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    total_people_detected = people_count  

cap.release()
cv2.destroyAllWindows()

print(f"Overall People Detected: {total_people_detected}")
