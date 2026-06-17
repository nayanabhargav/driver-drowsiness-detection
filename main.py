import cv2
import mediapipe as mp

from src.detector import DrowsinessDetector
from src.alarm import AlarmPlayer
from src.utils import draw_text_box

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 30
CAMERA_INDEX = 0
ALARM_PATH = "assets/alarm.wav"


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Increase camera resolution so 2 faces can appear clearly
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = DrowsinessDetector(
        ear_threshold=EAR_THRESHOLD,
        consecutive_frames=CONSECUTIVE_FRAMES,
        max_faces=2
    )

    alarm = AlarmPlayer(ALARM_PATH)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,          # Important for detecting multiple faces
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:

        print("Camera started.")
        print("Two-person drowsiness detection started.")
        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status_text = "ACTIVE"
            status_color = (0, 255, 0)
            any_person_drowsy = False
            detected_faces = 0

            if results.multi_face_landmarks:
                detected_faces = len(results.multi_face_landmarks)

                for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
                    if face_index >= 2:
                        break

                    is_drowsy, ear, face_box = detector.detect(
                        frame=frame,
                        face_landmarks=face_landmarks,
                        width=w,
                        height=h,
                        face_index=face_index
                    )

                    x_min, y_min, x_max, y_max = face_box

                    if is_drowsy:
                        any_person_drowsy = True
                        person_status = "DROWSY"
                        box_color = (0, 0, 255)
                    else:
                        person_status = "ACTIVE"
                        box_color = (0, 255, 0)

                    # Draw face rectangle
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

                    # Show person status near face
                    cv2.putText(
                        frame,
                        f"Person {face_index + 1}: {person_status}",
                        (x_min, max(y_min - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2
                    )

                    # Show EAR value on screen
                    cv2.putText(
                        frame,
                        f"Person {face_index + 1} EAR: {ear:.2f}",
                        (20, 90 + face_index * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2
                    )

                # Reset counters for missing faces
                detector.reset_missing_counters(detected_faces)

                if any_person_drowsy:
                    status_text = "DROWSINESS ALERT!"
                    status_color = (0, 0, 255)
                    alarm.play()
                else:
                    alarm.stop()

            else:
                status_text = "FACE NOT DETECTED"
                status_color = (0, 255, 255)
                detector.reset_all_counters()
                alarm.stop()

            # Show total detected faces
            cv2.putText(
                frame,
                f"Faces Detected: {detected_faces}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            draw_text_box(frame, status_text, status_color)

            cv2.imshow("Two-Person AI-Based Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()