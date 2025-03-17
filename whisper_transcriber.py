from faster_whisper import WhisperModel
import numpy as np

class WhisperTranscriber:
    def __init__(self, model_size="medium", device="cuda", compute_type="int8"):
        """
        Using faster-whisper to initialize Whisper model
        :param model_size: choose model size, such as "small", "medium", "large-v2"
        :param device: choose device, such as "cuda", "cpu"
        :param compute_type: choose compute type, such as "int8", "float16", "float32"
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def process_audio_chunk(self, audio_chunk, beam_size=5, temperature=0) -> list:
        """
        Transcribe audio chunk using Whisper model
        :param audio_chunk: np.array format of audio data (already normalized)
        :param beam_size: beam search size, affecting accuracy and speed
        :param temperature: parameter affecting model exploration, 0 means most certain result
        :return: transcribed text
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None

        segments, _ = self.model.transcribe(audio_chunk, beam_size=beam_size, temperature=temperature)

        transcribed_text = [segment.text for segment in segments]
        return transcribed_text