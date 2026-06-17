import cv2


def draw_text_box(frame, text, color):
    """
    Draws status text on camera frame.
    """

    x, y, w, h = 15, 20, 430, 45

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)

    cv2.putText(
        frame,
        f"Status: {text}",
        (x + 10, y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )