import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

last_pose_landmark_data = None

def within_range(value, target, tolerance):
    return target - tolerance <= value <= target + tolerance

def find_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)



if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

buffer = 0

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break

        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = pose.process(rgb_frame)
    hand_result = hands.process(rgb_frame)
    
    
    if result.pose_landmarks or hand_result.multi_hand_landmarks:
        #for pose_landmarks in result.multi_pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hand_result.multi_hand_landmarks:
            for hand_landmark in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            
            
        this_pose_landmark_data = []
        
        for lm in result.pose_landmarks.landmark[12:23:2]:
            if buffer % 10 == 0 and lm.visibility > .75:
                this_pose_landmark_data.append((lm.x,  lm.y,  lm.z))
            elif buffer % 10 == 0:
                this_pose_landmark_data.append((0,0,0))
                
       
                
        if this_pose_landmark_data:  
            #h_L1 = find_distance(this_pose_landmark_data[0], this_pose_landmark_data[2])
            print(f"Distance from base to end effector: {math.sqrt((this_pose_landmark_data[2][0] - this_pose_landmark_data[0][0]) ** 2 + (this_pose_landmark_data[2][1] - this_pose_landmark_data[0][1]) ** 2)}")
            print()
    buffer += 1
    
            
            
    
    
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
