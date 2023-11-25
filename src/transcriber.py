from faster_whisper import WhisperModel
import numpy as np
from context import Context

class Transcriber:

    def __init__(self, ctx: Context, model_size='small') -> None:
        self.model = WhisperModel(model_size, device="cpu", compute_type="float32")
        self.ctx = ctx

    def transcribe_segments(self, beam_size=5) -> None:
        """
        Transcribe audio buffer and return segments and info
        Requires that audio buffer is a numpy array of floats with a sample rate of 16000 
        """
        with self.ctx.file_mutex:
            segments, info = self.model.transcribe(self.ctx.transcription_buffer, beam_size=beam_size)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            self.ctx.all_segments.append(segment.text)