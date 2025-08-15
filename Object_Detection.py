import cv2

cap = cv2.VideoCapture(0) # This is video capture object

# Set width and height (try HD: 1280x720 or Full HD: 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur  reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to segment object from background
    _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (object outline)
    contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500: # skip tiny noise

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            # Draw rotated rectangle
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # Extract dimensions
            width, height = rect[1]
            angle = rect[2]

            cv2.putText(frame, f"L: {width:.1f}px, H: {height:.1f}px", 
                        (box[0][0], box[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    frame = cv2.resize(frame, (1200, 800))
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()