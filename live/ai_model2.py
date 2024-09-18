import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import moviepy.editor as mp
import numpy as np
from tensorflow.keras.models import load_model


def analyze_image2(image_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    def load_detection_model(model_path):
        detection_model = model_path
        return detection_model

    def draw_bounding_box(face_coordinates, image_array, color, img_width, img_height):
        x, y, w, h = (int(face_coordinates.location_data.relative_bounding_box.xmin * img_width),
                      int(face_coordinates.location_data.relative_bounding_box.ymin * img_height),
                      int(face_coordinates.location_data.relative_bounding_box.width * img_width),
                      int(face_coordinates.location_data.relative_bounding_box.height * img_height))
        cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

    def apply_offsets(face_coordinates, offsets, img_width, img_height):
        x, y, width, height = (int(face_coordinates.location_data.relative_bounding_box.xmin * img_width),
                               int(face_coordinates.location_data.relative_bounding_box.ymin * img_height),
                               int(face_coordinates.location_data.relative_bounding_box.width * img_width),
                               int(face_coordinates.location_data.relative_bounding_box.height * img_height))

        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    def draw_text(coordinates, image_array, text, color, img_width, img_height, x_offset=0, y_offset=0, font_scale=0.5,
                  thickness=2):
        x, y = (int(coordinates.location_data.relative_bounding_box.xmin * img_width),
                int(coordinates.location_data.relative_bounding_box.ymin * img_height))
        cv2.putText(image_array, text, (x + x_offset, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    thickness, cv2.LINE_AA)

    def preprocess_input(x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    image_path = image_path  # 사진 파일 이름
    detection_model_path = mp.solutions.face_detection

    emotion_model_path = 'emotion_model_InceptionV3.h5'
    emotion_labels = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'sad', 4: 'surprise', 5: 'neutral'}
    font = cv2.FONT_HERSHEY_SIMPLEX

    emotion_offsets = (0, 0)

    # face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]

    rgb_image = cv2.imread(image_path)
    gray_image = cv2.imread(image_path, 0)

    def face_detect(image):
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Draw face detections of each face.
            if not results.detections:
                print("Face not found in image")
            else:
                print('Found {} faces.'.format(len(results.detections)))

                annotated_image = image.copy()

            return results.detections

    def highest_emotion(positive, neutral, negative):
        if positive == max([positive, neutral, negative]):
            return 'positive'
        elif neutral == max([positive, neutral, negative]):
            return 'neutral'
        else:
            return 'negative'

    def engagement_score(scores):
        if ((scores[5] > 0.6) | (scores[2] > 0.5) | (scores[4] > 0.6) | (scores[0] > 0.2) | (scores[1] > 0.3) | (
                scores[3] > 0.3)):
            return ((scores[0] * 0.25) + (scores[1] * 0.3) + (scores[2] * 0.6) + (scores[3] * 0.3) + (
                    scores[4] * 0.6) + (
                            scores[5] * 0.9))
        else:
            return 0

    positives = []
    neutrals = []
    negatives = []
    engagements = []
    faces = face_detect(rgb_image)

    for face_coordinates in faces:
        print(face_coordinates)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets, rgb_image.shape[1], rgb_image.shape[0])
        gray_face = gray_image[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, (emotion_target_size))

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        engagement = engagement_score(emotion_classifier.predict(gray_face)[0])
        positive = emotion_classifier.predict(gray_face)[0][2] + emotion_classifier.predict(gray_face)[0][4]
        neutral = emotion_classifier.predict(gray_face)[0][5]
        negative = emotion_classifier.predict(gray_face)[0][0] + emotion_classifier.predict(gray_face)[0][1] + \
                   emotion_classifier.predict(gray_face)[0][3]
        positives.append(positive)
        neutrals.append(neutral)
        negatives.append(negative)
        engagements.append(engagement)
        emotion_text = highest_emotion(positive, neutral, negative)
        color = (0, 255, 255)

        draw_bounding_box(face_coordinates, rgb_image, color, rgb_image.shape[1], rgb_image.shape[0])
        draw_text(face_coordinates, rgb_image, emotion_text, color, rgb_image.shape[1], rgb_image.shape[0], -20, -20,
                  0.7, 2)

        plt.imshow(rgb_image)
        cv2.imwrite('result_emotion_image.jpg', rgb_image)

        def calculate_percentage(emotion):
            if len(emotion) == 0:
                return 0
            else:
                return int(round(sum(emotion) / len(emotion), 2) * 100)

        positive = calculate_percentage(positives)
        neutral = calculate_percentage(neutrals)
        negative = calculate_percentage(negatives)
        concentration = calculate_percentage(engagements)
        results = {
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "concentration": concentration,
        }

        return results
