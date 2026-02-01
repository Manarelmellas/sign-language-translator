import cv2
from hand_detection import detect_hand

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, hand_coords_list = detect_hand(frame)

    # Afficher les coordonnées dans la console pour tester
    for i, hand_coords in enumerate(hand_coords_list):
        print(f"Hand {i+1}: {hand_coords}")

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
