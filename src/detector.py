import cv2
import numpy as np


class DrowsinessDetector:
    """
    Detects drowsiness using Eye Aspect Ratio.
    Supports two-person detection separately.
    """

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold=0.25, consecutive_frames=30, max_faces=2):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.max_faces = max_faces

        # Separate drowsiness counter for each person
        self.frame_counters = [0] * max_faces

    def _landmark_to_point(self, landmark, width, height):
        return int(landmark.x * width), int(landmark.y * height)

    def _eye_aspect_ratio(self, eye_points):
        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

        if horizontal == 0:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def _draw_eye_landmarks(self, frame, points, color):
        for point in points:
            cv2.circle(frame, point, 3, color, -1)

        for i in range(len(points)):
            cv2.line(frame, points[i], points[(i + 1) % len(points)], color, 1)

    def _get_face_box(self, landmarks, width, height):
        x_points = []
        y_points = []

        for landmark in landmarks:
            x_points.append(int(landmark.x * width))
            y_points.append(int(landmark.y * height))

        x_min = max(min(x_points) - 20, 0)
        y_min = max(min(y_points) - 20, 0)
        x_max = min(max(x_points) + 20, width)
        y_max = min(max(y_points) + 20, height)

        return x_min, y_min, x_max, y_max

    def detect(self, frame, face_landmarks, width, height, face_index=0):
        landmarks = face_landmarks.landmark

        left_eye_points = [
            self._landmark_to_point(landmarks[i], width, height)
            for i in self.LEFT_EYE
        ]

        right_eye_points = [
            self._landmark_to_point(landmarks[i], width, height)
            for i in self.RIGHT_EYE
        ]

        # Different eye landmark color for each person
        if face_index == 0:
            eye_color = (0, 255, 255)      # Yellow
        else:
            eye_color = (255, 0, 255)      # Purple

        self._draw_eye_landmarks(frame, left_eye_points, eye_color)
        self._draw_eye_landmarks(frame, right_eye_points, eye_color)

        left_ear = self._eye_aspect_ratio(left_eye_points)
        right_ear = self._eye_aspect_ratio(right_eye_points)

        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < self.ear_threshold:
            self.frame_counters[face_index] += 1
        else:
            self.frame_counters[face_index] = 0

        is_drowsy = self.frame_counters[face_index] >= self.consecutive_frames

        face_box = self._get_face_box(landmarks, width, height)

        return is_drowsy, avg_ear, face_box

    def reset_missing_counters(self, detected_faces):
        for i in range(detected_faces, self.max_faces):
            self.frame_counters[i] = 0

    def reset_all_counters(self):
        for i in range(len(self.frame_counters)):
            self.frame_counters[i] = 0
