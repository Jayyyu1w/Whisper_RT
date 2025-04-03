import pyaudio
import torch
import numpy as np
from threading import Thread
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

from whisper_transcriber import WhisperTranscriber

class AudioRecorder:
    def __init__(self, rate=16000, chunk_size=8000, channels=1, threshold_seconds=3):
        self._rate = rate
        self._chunk_size = chunk_size
        self._channels = channels
        self._threshold_seconds = threshold_seconds
        self._audio_format = pyaudio.paInt16  # 16-bit PCM

        self._audio = pyaudio.PyAudio()
        self._stream = None
        self._audio_chunk = list()  # 使用 list 來儲存音訊資料

        self.recording_thread = None  # 錄音執行緒

    def list_audio_devices(self):
        """列出所有可用的音訊設備"""
        print("Usable audio devices:")
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            print(f"Device index: {i}, Device name: {info['name']}")

    def get_default_device(self):
        """獲取預設音訊輸入裝置"""
        default_device_info = self._audio.get_default_input_device_info()
        print(f"Current use device: {default_device_info['name']}")
        return default_device_info

    def start_recording(self):
        """開始錄音"""
        if self._audio is None:
            self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=self._audio_format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            stream_callback=self._put_audio_chunk
        )
        print("Open stream successfully.")

    def _put_audio_chunk(self, chunk):
        """Callback function to put audio chunk into list"""
        self._audio_chunk.append(chunk)  # 將音訊資料放入 list
        if len(self._audio_chunk) >= self._threshold_seconds * 2:  # 每秒 2 個 chunk
            audio_data = self._audio_chunk.copy()
            self._audio_chunk.clear()
            
            self._transcribe_audio(audio_data)

        return None, pyaudio.paContinue

    def _transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper model"""
        # 這裡可以加入 Whisper 模型的轉錄邏輯
        # 例如：使用 WhisperModel 進行轉錄
        # 這裡假設有一個 WhisperTranscriber 類別可以使用
        transcriber = WhisperTranscriber()
        transcriber_thread = Thread(
            target=transcriber.process_audio_chunk,
            args=(audio_data, 5, 0),
            daemon=True
        )
        transcriber_thread.start()
        pass

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio:  
            self._audio.terminate()
            self._audio = None
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