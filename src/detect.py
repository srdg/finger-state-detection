import cv2
import math
import mediapipe as mp
 
def getAngle(a, b, c):
    '''Get the angle made by the lines between coordinates (a,b) and (b,c).'''
    ang = math.degrees(math.atan2(c.y-b.y, c.x-b.x) - math.atan2(a.y-b.y, a.x-b.x))
    return ang + 360 if ang < 0 else ang

def getFingerState(joint_angle1, joint_angle2):
    '''Return a string boolean indicating if finger is open or closed 
       based on the value of the angles made by the joints.'''
    return str(90 < joint_angle1 < 190 and 150 < joint_angle2 < 190)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands    

# For webcam input:
cap = cv2.VideoCapture(0)
drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128))
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    fingers = [False, False, False, False, False]
    finger_labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      landMark_idx = 1
      for hand_landmarks in results.multi_hand_landmarks:
        # Get the angles between the joints and store them
        landmark = hand_landmarks.landmark
        hand_landmark = mp_hands.HandLandmark
        a, coord_joint1 = [], []
        b, coord_joint2 = [], []
        for _ in range(5):
          A = getAngle(landmark[landMark_idx], landmark[landMark_idx+1], landmark[landMark_idx+2])
          B = getAngle(landmark[landMark_idx+1], landmark[landMark_idx+2], landmark[landMark_idx+3])
          a.append(int(A))
          coord_joint1.append((int(landmark[landMark_idx+1].x*image.shape[1]), int(landmark[landMark_idx+1].y*image.shape[0])))
          b.append(int(B))
          coord_joint2.append((int(landmark[landMark_idx+2].x*image.shape[1]), int(landmark[landMark_idx+2].y*image.shape[0])))
          landMark_idx+=4
          landMark_idx = landMark_idx%21
      # Get the state of fingers (i.e. open or closed)
      for idx in range(5):
          fingers[idx]=getFingerState(a[idx],b[idx])      
      # Parameters for writing on image
      offset = 10
      height_separation = 15
      # Write state of fingers on image frame
      for idx in range(len(finger_labels)):
           string = finger_labels[idx]+': '+fingers[idx]
           co_ord = (image.shape[1]-100, offset)
           cv2.putText(image, string, co_ord, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230,230, 250))
           offset += height_separation
      # Write angles beside the joints
      for index in range(len(a)):
        cv2.putText(image, str(a[index]), coord_joint1[index], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,0,128))
        cv2.putText(image, str(b[index]), coord_joint2[index], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,0,128))
      # Draw the detected landmarks
        mp_drawing.draw_landmarks(
            image, hand_landmarks,mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
