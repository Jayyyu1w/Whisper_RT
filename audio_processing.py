import pyaudio
import torch
import numpy as np
import queue
from queue import Queue
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

class AudioRecorder:
    def __init__(self, rate=16000, chunk_size=2000, channels=1, threshold_seconds=3):
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.threshold_seconds = threshold_seconds

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = Queue()

    def list_audio_devices(self):
        """List all available audio devices."""
        print("Usable audio devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"Device index: {i}, Device name: {info['name']}")

    def get_default_device(self):
        """Get the default audio input device."""
        default_device_info = self.audio.get_default_input_device_info()
        print(f"Current use device: {default_device_info['name']}")
        return default_device_info

    def start_recording(self):
        """Start recording audio."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        print("Start recording...")

    def record_audio(self):
        """Record audio and put it into the audio queue."""
        data = self.stream.read(self.chunk_size)
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        self.audio_queue.put(audio_chunk)

    def process_audio_queue(self):
        """Process the audio queue and return the accumulated audio."""
        accumulated_audio = []
        while not self.audio_queue.empty():
            accumulated_audio.append(self.audio_queue.get())
        return np.concatenate(accumulated_audio) if accumulated_audio else None

    def stop_recording(self):
        """Stop recording audio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Recording stopped.")


class SileroVAD:
    def __init__(self, mode=3):
        """
        Initialize SileroVAD with mode
        """
        self.vad = load_silero_vad()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if the audio chunk contains speech
        :param audio_chunk: np.ndarray, audio chunk
        :return: bool, True if the audio chunk contains speech, False otherwise
        """
        # audio_chunk from ndarray to tensor
        audio_chunk = torch.tensor(audio_chunk, dtype=torch.float32)
        
        speech_timestamps = get_speech_timestamps(
            audio_chunk,
            self.vad,
            return_seconds=True,
        )
        return len(speech_timestamps) > 0