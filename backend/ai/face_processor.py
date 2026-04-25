import cv2
import numpy as np
import base64
import mediapipe as mp

mp_face = mp.solutions.face_detection.FaceDetection(0.6)


def encode_image(img):

    _, buffer = cv2.imencode(".jpg", img)

    return base64.b64encode(buffer).decode("utf-8")


def process_face_image(image_bytes):

    np_arr = np.frombuffer(image_bytes, np.uint8)

    if np_arr is None or len(np_arr) == 0:
        return None, "Invalid or empty image file"
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Failed to decode image"

    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = mp_face.process(rgb)

    if not results.detections:
        return None, "No face detected"

    detection = results.detections[0]

    box = detection.location_data.relative_bounding_box

    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    annotated = img.copy()

    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    face = img[y:y+bh, x:x+bw]

    if face.size == 0:
        return None, "Face crop failed"

    fh, fw = face.shape[:2]

    crops = {
        "forehead": face[0:int(fh*0.25), int(fw*0.25):int(fw*0.75)],
        "left_cheek": face[int(fh*0.35):int(fh*0.65), 0:int(fw*0.35)],
        "right_cheek": face[int(fh*0.35):int(fh*0.65), int(fw*0.65):fw],
        "nose": face[int(fh*0.35):int(fh*0.65), int(fw*0.35):int(fw*0.65)],
        "chin": face[int(fh*0.70):fh, int(fw*0.30):int(fw*0.70)],
        "left_eye": face[int(fh*0.20):int(fh*0.35), int(fw*0.15):int(fw*0.35)],
        "right_eye": face[int(fh*0.20):int(fh*0.35), int(fw*0.65):int(fw*0.85)],
    }

    encoded_crops = {}

    for k, v in crops.items():
        if v.size == 0:
            continue
        encoded_crops[k] = encode_image(v)

    return {
        "crops": crops,
        "encoded_crops": encoded_crops,
        "annotated_image": encode_image(annotated)
    }, None