from faster_whisper import WhisperModel
import numpy as np
from context import Context
import threading


class Transcriber:
    def __init__(self, ctx: Context, model_size="small") -> None:
        self.model = WhisperModel(
            "models/whisper-small-pt-cv11-v7", device="cpu", compute_type="float32"
        )
        self.ctx = ctx
        self.duration = 0.0
        self.is_transcribing = False

    def reset_duration(self) -> None:
        self.duration = 0.0

    def replace_model(self, model_name: str) -> None:
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="float32",
        )
        print(f"Loaded model {model_name}")

    def transcribe_segments(self, beam_size=5) -> None:
        self.ctx.transcriber_lock.acquire()
        self.is_transcribing = True
        """
        Transcribe audio buffer and return segments and info
        Requires that audio buffer is a numpy array of floats with a sample rate of 16000
        """
        while True:
            try:
                item = self.ctx.transcription_queue.get(block=True, timeout=1.0)
            except:
                if not self.ctx.is_recording:
                    self.ctx.transcriber_lock.release()
                    self.is_transcribing = False
                    return
                continue
            else:
                segs, info = self.model.transcribe(item, beam_size=beam_size)

                print(
                    "Detected language '%s' with probability %f"
                    % (info.language, info.language_probability)
                )

                self.duration += self.ctx.transcription_buffer_duration

                if info.language_probability > 0.85:
                    for segment in segs:
                        print(
                            "[%.2fs -> %.2fs] %s"
                            % (segment.start, segment.end, segment.text)
                        )
                        new_seg = Segment(
                            info.language,
                            info.language_probability,
                            self.duration + segment.start,
                            self.duration + segment.end,
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
        return f"({self.language}, {self.language_prob:.2f}) | [{self.start:4.2f}s -> {self.end:4.2f}s]: {self.text.strip()}"


class FileTranscriber:
    def __init__(
        self,
        model_size="small",
        input_filename="output.wav",
        output_filename="output.txt",
    ) -> None:
        self.model_size = model_size
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.model = WhisperModel(model_size, device="cpu", compute_type="float32")
        self.segments: list[Segment] = []
        self.transcribing = False
        self.model_lock = threading.Lock()

    def change_input_filename(self, input_filename: str) -> None:
        self.input_filename = input_filename

    def change_output_filename(self, output_filename: str) -> None:
        self.output_filename = output_filename

    def transcribe_file(self, bean_size=5) -> None:
        self.transcribing = True
        with self.model_lock:
            segs, info = self.model.transcribe(self.input_filename, beam_size=bean_size)
        self.transcribing = False

        for segment in segs:
            new_seg = Segment(
                info.language,
                info.language_probability,
                segment.start,
                segment.end,
                segment.text,
            )
            self.segments.append(new_seg)

    def save_transcription(self) -> None:
        with open(self.output_filename, "w") as f:
            for segment in self.segments:
                f.write(str(segment) + "\n")
    
    def change_model(self, model_name: str) -> None:
        with self.model_lock:
            self.model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="float32",
            )
        print(f"Loaded model {model_name}")