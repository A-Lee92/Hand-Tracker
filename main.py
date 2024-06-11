import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

last_hand_landmark_data = None

def within_range(value, target, tolerance):
    return target - tolerance <= value <= target + tolerance



if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break

        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb_frame)
    
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            
            this_hand_landmark_data = []
            for lm in hand_landmarks.landmark:
                this_hand_landmark_data.append((lm.x, lm.y, lm.z))
            if last_hand_landmark_data:
                if within_range(sum(this_hand_landmark_data[-1]), sum(last_hand_landmark_data[-1]), 1.5):
                    print(this_hand_landmark_data[-1])
                
            last_hand_landmark_data = this_hand_landmark_data
            
    
        
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
