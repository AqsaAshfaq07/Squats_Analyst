import cv2, os, time
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point  # hip
    b = np.array(b)  # Mid point  # knee
    c = np.array(c)  # End point  # ankle
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Create a unique directory to store frames for each trial
def create_unique_directory(base_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    trial_dir = os.path.join(base_dir, f"squats_{timestamp}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

# Function to get the coordinates of a specific landmark
def get_landmark_coords(landmarks, landmark_name):
    landmark = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return [landmark.x, landmark.y]

# Function to check the current position (Standing or Lowered)
def check_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')

    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    ankle_angle = calculate_angle(knee, ankle, (ankle[0], ankle[1] - 1))  # Approximate vertical line

    if hip_angle > 160 and knee_angle > 160:
        return "Standing"
    elif hip_angle < 120 and knee_angle < 120:
        return "Lowered"
    else:
        return "Standing"

# Function to evaluate standing position
def evaluate_standing_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')

    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = 180

    feedback = "Good posture"
    if knee_angle < 170:
        feedback = "Straighten your knees"
    elif knee_angle > 190:
        feedback = "Relax your knees"

    return feedback, hip_angle, knee_angle

# Function to evaluate lowered position
def evaluate_lowered_position(landmarks):
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')

    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    feedback = "Good posture"
    if hip_angle < 70:
        feedback = "Lower your hips more"
    elif hip_angle > 120:
        feedback = "Raise your hips a bit"

    if knee_angle < 70:
        feedback += " and bend your knees more"
    elif knee_angle > 120:
        feedback += " and bend your knees less"

    return feedback, hip_angle, knee_angle

# Specify the video file path here
video_file_path = 'files/squatt.mp4'

# Main function to process the video file
def process_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    
    squat_count = 0
    last_position = None
    frameno = 0
    dir = "frames/squats"
    trial_dir = create_unique_directory(dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            position = check_position(landmarks)
            feedback = ""  # Initialize feedback variable

            if position == "Standing":
                feedback, hip_angle, knee_angle  = evaluate_standing_position(landmarks)
            elif position == "Lowered":
                feedback, hip_angle, knee_angle = evaluate_lowered_position(landmarks)
            else: 
                feedback = "Transitioning"
            
            if last_position == "Lowered" and position == "Standing":
                squat_count += 1

            last_position = position
            frame = cv2.resize(frame, (900, 700))

            cv2.putText(frame, f'Squats: {squat_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Position: {position}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, feedback, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Hip Angle: {hip_angle}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Knee Angle: {knee_angle}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            name = os.path.join(trial_dir, f"{frameno}.jpg")
            print('new frame captured...' + name)
            cv2.imwrite(name, frame)
            frameno += 1

        cv2.imshow('Exercise Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the specified video file
process_video(0)