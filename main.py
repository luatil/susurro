#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from imgui.integrations.pygame import PygameRenderer
import OpenGL.GL as gl
import imgui
import pygame
import sys
import numpy as np
import pyaudio
import wave
import threading
from faster_whisper import WhisperModel
import librosa

is_recording = False
data_chunk  = np.zeros(1024, dtype=np.float32) 

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000 
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "file.wav"
BUFFER_DURATION = 3.0 # in seconds
MODEL_SIZE = "small"

file_mutex = threading.Lock()
model = WhisperModel(MODEL_SIZE, device="auto", compute_type="float32")
audio_buffer = np.zeros(int(RATE * BUFFER_DURATION), dtype=np.float32)
global_segments = []
transcription_buffer = None

def transcribe_audio():
    global global_segments
    global audio_buffer

    with file_mutex:
        print("Start trascription")
        segments, info = model.transcribe(transcription_buffer, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        global_segments.append(segment.text)
    

# Function to record audio
def record_audio():

    global audio_buffer
    global data_size
    global transcription_buffer

    with file_mutex:
        audio_buffer = np.zeros_like(audio_buffer)

        audio = pyaudio.PyAudio()

        # Start recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        frames = []

        while is_recording:
            data = stream.read(CHUNK)
            frames.append(data)
            buf = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_buffer = np.roll(audio_buffer, -len(buf))
            audio_buffer[-len(buf):] = buf

        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        transcription_buffer = np.frombuffer(b''.join(frames), dtype='int16').astype(np.float32) / 32768.0
        transcription_buffer = librosa.resample(transcription_buffer, orig_sr=RATE, target_sr=16000)

        # Save the recorded data as a WAV file
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        print("End of recording")



def main():
    global is_recording 

    time = 0.0

    pygame.init()
    size = 800, 600

    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)

    imgui.create_context()
    impl = PygameRenderer()

    io = imgui.get_io()
    io.display_size = size

    yscale = 5000.0
    min_yscale = 1000.0
    max_yscale = 10000.0

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            impl.process_event(event)
        impl.process_inputs()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    sys.exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Sine Wave", True)
        chaged, yscale = imgui.slider_float("Y Scale", yscale, min_yscale, max_yscale)
        imgui.plot_lines("Sine", audio_buffer, scale_min=-yscale, scale_max=+yscale, graph_size=(0, 200))
        imgui.end()

        imgui.begin("Audio Recorder", True)
        if imgui.button("Record"):
            if not is_recording:
                is_recording = True
                threading.Thread(target=record_audio, daemon=True).start()

        imgui.same_line()

        if imgui.button("Stop Recording"):
            is_recording = False
            threading.Thread(target=transcribe_audio, daemon=True).start()
        imgui.end()

        imgui.begin("Transcription", True)
        for segment in global_segments:
            imgui.text(segment)
        imgui.end()

        # note: cannot use screen.fill((1, 1, 1)) because pygame's screen
        #       does not support fill() on OpenGL sufraces
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())

        pygame.display.flip()


if __name__ == "__main__":
    main()
