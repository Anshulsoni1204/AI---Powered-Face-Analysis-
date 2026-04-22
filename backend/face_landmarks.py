import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Initialize Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_face_landmarks(image):
    """
    Input: BGR image (OpenCV format)
    Output: list of landmarks or None
    """

    # Convert BGR → RGB (MediaPipe requirement)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    landmarks = []

    # Extract (x, y, z) normalized coordinates
    for landmark in results.multi_face_landmarks[0].landmark:
        landmarks.append({
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z
        })

    return landmarks