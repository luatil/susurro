from faster_whisper import WhisperModel
import numpy as np
from context import Context


class Transcriber:
    def __init__(self, ctx: Context, model_size="small") -> None:
        self.model = WhisperModel(
            "models/whisper-small-pt-cv11-v7", device="cpu", compute_type="float32"
        )
        self.ctx = ctx

    def transcribe_segments(self, beam_size=5) -> None:
        """
        Transcribe audio buffer and return segments and info
        Requires that audio buffer is a numpy array of floats with a sample rate of 16000
        """
        while True:
            item = self.ctx.transcription_queue.get(block=True)
            if item is None:
                continue
            else:
                segs, info = self.model.transcribe(item, beam_size=beam_size)

                print(
                    "Detected language '%s' with probability %f"
                    % (info.language, info.language_probability)
                )

                if info.language_probability > 0.85:
                    for segment in segs:
                        print(
                            "[%.2fs -> %.2fs] %s"
                            % (segment.start, segment.end, segment.text)
                        )
                        new_seg = Segment(
                            info.language,
                            info.language_probability,
                            segment.start,
                            segment.end,
                            segment.text,
                        )
                        self.ctx.all_segments.append(new_seg)


class Segment:
    def __init__(
        self, language: str, language_prob: float, start: float, end: float, text: str
    ) -> None:
        self.language = language
        self.language_prob = language_prob
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self) -> str:
        return f"({self.language}, {self.language_prob:.2f}) | [{self.start:.2f}s -> {self.end:.2f}s]: {self.text.strip()}"
