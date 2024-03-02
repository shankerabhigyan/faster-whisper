import logging
import os
from functools import lru_cache
from typing import Any

import librosa


@lru_cache
def load_audio(file_name: str) -> Any:
    """loads the given audio file

    Args:
        file_name (str): audio file path

    Returns:
        Any: audio data
    """
    audio, _ = librosa.load(file_name, sr=16000)
    return audio

def load_audio_chunk(file_name: str, beg: int, end: int) -> Any:
    """loads audio chunk based on beg and end 

    Args:
        file_name (str): audio file path
        beg (int): beginning of chunk
        end (int): ending of chunk

    Returns:
        Any: audio chunk
    """
    audio = load_audio(file_name)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


def test_service(demo_audio_path: str, asr: Any):
    if os.path.exists(demo_audio_path):
        try:
            # load the audio into the LRU cache before we start the timer
            audio = load_audio_chunk(demo_audio_path,0,1)
            # warm up the ASR, because the very first transcribe takes much more time than the other
            text = asr.transcribe(audio)
            logging.info(f"Test successful - {text}")
        except Exception as e: 
            logging.info(f"ERROR {e} in testing service")
            raise e
    