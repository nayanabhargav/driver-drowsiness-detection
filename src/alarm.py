from pathlib import Path
import pygame


class AlarmPlayer:
    """
    Plays alarm sound when drowsiness is detected.
    """

    def __init__(self, alarm_path):
        self.alarm_path = Path(alarm_path)
        self.is_playing = False

        pygame.mixer.init()

        if not self.alarm_path.exists():
            print(f"Warning: Alarm file not found: {self.alarm_path}")
            print("Alarm will not play until the file is added.")

    def play(self):
        if not self.alarm_path.exists():
            return

        if not self.is_playing:
            pygame.mixer.music.load(str(self.alarm_path))
            pygame.mixer.music.play(-1)
            self.is_playing = True

    def stop(self):
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
