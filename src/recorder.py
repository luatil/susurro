from context import Context
import pyaudio
import numpy as np
import librosa
import wave


class Recorder:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.audio_recording = None

    def record_audio(self):
        """
        Record audio and store in context
        """
        with self.ctx.file_mutex:
            self.audio_io = pyaudio.PyAudio()

            stream = self.audio_io.open(
                format=self.ctx.audio_recording_format,
                channels=self.ctx.channels,
                rate=self.ctx.sr,
                input=True,
                frames_per_buffer=self.ctx.chunk_size,
            )

            frames = []

            read_length = 0
            temp_frames = []

            while self.ctx.is_recording:
                data = stream.read(self.ctx.chunk_size)
                frames.append(data)
                temp_frames.append(data)
                read_length += len(data)

                if read_length > self.ctx.transcription_length:
                    buf = (
                        np.frombuffer(b"".join(temp_frames), dtype=np.int16).astype(
                            np.float32
                        )
                        / 32768.0
                    )
                    print(buf.sum())
                    buf = librosa.resample(buf, orig_sr=self.ctx.sr, target_sr=16000)
                    self.ctx.transcription_queue.put(buf)
                    temp_frames = []
                    read_length = 0

                buf = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                self.ctx.audio_buffer = np.roll(self.ctx.audio_buffer, -len(buf))
                self.ctx.audio_buffer[-len(buf) :] = buf

            stream.stop_stream()
            stream.close()
            self.audio_io.terminate()

            self.ctx.transcription_buffer = (
                np.frombuffer(b"".join(frames), dtype="int16").astype(np.float32)
                / 32768.0
            )
            self.ctx.transcription_buffer = librosa.resample(
                self.ctx.transcription_buffer, orig_sr=self.ctx.sr, target_sr=16000
            )
            # self.ctx.transcription_queue.put(self.ctx.transcription_buffer)

            self.audio_recording = b"".join(frames)

    def save_audio(self, filename):
        with self.ctx.file_mutex:
            waveFile = wave.open(filename, "wb")
            waveFile.setnchannels(self.ctx.channels)
            waveFile.setsampwidth(
                pyaudio.get_sample_size(self.ctx.audio_recording_format)
            )
            waveFile.setframerate(self.ctx.sr)
            waveFile.writeframes(self.audio_recording)
            waveFile.close()
