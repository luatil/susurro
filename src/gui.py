import pygame
import imgui
from imgui.integrations.pygame import PygameRenderer
from context import Context
import sys
import threading
from recorder import Recorder
from transcriber import Transcriber
import OpenGL.GL as gl

class GUI:
    def __init__(self, ctx: Context, w=800, h=600) -> None:
        self.w = w
        self.h = h
        self.ctx = ctx

        # Start pygame
        pygame.init()
        size = self.w, self.h

        pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)

        imgui.create_context()
        self.impl = PygameRenderer()

        io = imgui.get_io()
        io.display_size = size

        # Menu bar
        self.menu = MenuBar(ctx)

        # Sine plot
        self.sine_plot = SinePlot(ctx)

        # Recorder
        self.recorder = AudioRecorderWindow(ctx)

        # Transcription Window
        self.transcription_window = TranscriptionWindow(ctx)

    def render(self) -> None:

        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                self.impl.process_event(event)
            self.impl.process_inputs()

            imgui.new_frame()

            # Render Menu
            self.menu.render()

            # Render Sine plot
            self.sine_plot.render()

            # Render recorder
            self.recorder.render()

            # Render transcription window
            self.transcription_window.render()

            # note: cannot use screen.fill((1, 1, 1)) because pygame's screen
            #       does not support fill() on OpenGL sufraces
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            imgui.render()
            self.impl.render(imgui.get_draw_data())

            pygame.display.flip()

        
class MenuBar:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def render(self) -> None:
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    sys.exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()


class SinePlot:
    def __init__(self, ctx: Context, yscale=5000.0, min_yscale=1000.0, max_yscale=10000.0) -> None:
        self.ctx = ctx
        self.yscale = yscale
        self.min_yscale = min_yscale
        self.max_yscale = max_yscale

    def render(self) -> None:
        imgui.begin("Sine Wave", True)
        changed, self.yscale = imgui.slider_float("Y Scale", self.yscale, self.min_yscale, self.max_yscale)
        imgui.plot_lines("Sine", self.ctx.audio_buffer, scale_min=-self.yscale, scale_max=+self.yscale, graph_size=(0, 200))
        imgui.end()

class AudioRecorderWindow:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.recorder = Recorder(ctx)
        self.transcriber = Transcriber(ctx)

    def render(self) -> None:
        imgui.begin("Audio Recorder", True)
        if imgui.button("Record"):
            if not self.ctx.is_recording:
                self.ctx.is_recording = True
                threading.Thread(target=self.recorder.record_audio, daemon=True).start()

        imgui.same_line()

        if imgui.button("Stop"):
            self.ctx.is_recording = False
            threading.Thread(target=self.transcriber.transcribe_segments, daemon=True).start()

        if imgui.button("Save"):
            threading.Thread(target=self.recorder.save_audio('output.wav'), daemon=True).start()

        imgui.end()

class TranscriptionWindow:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def render(self) -> None:
        imgui.begin("Transcription", True)
        for segment in self.ctx.all_segments:
            imgui.text(segment)
        imgui.end()