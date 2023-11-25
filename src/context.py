import threading
import numpy as np
import pyaudio


class Context:
    def __init__(self, sr=48000, buffer_duration=3.0, chunk_size=1024) -> None:
        self.sr = sr
        self.buffer_duration = buffer_duration
        self.channels = 1
        self.chunk_size = chunk_size
        self.file_mutex = threading.Lock()
        self.is_recording = False
        self.audio_buffer = np.zeros(int(sr * buffer_duration), dtype=np.float32)
        self.global_segments = []
        self.transcription_buffer = None

        self.all_segments = []
        self.current_segments = None
        self.all_info = []
        self.current_info = None

        self.audio_recording_format = pyaudio.paInt16
