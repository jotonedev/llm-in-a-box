import logging
import os
import threading
import time
from concurrent import futures
from typing import Callable

import speech_recognition as sr

from .llm_manager import LLMManager, LLMTask

__all__ = ["STTManager"]

logger = logging.getLogger("llmbox.stt")


class STTManager:
    _executor: futures.Executor
    _token: str
    _llm_manager: LLMManager

    def __init__(
        self,
        token: str,
        llm_manager: LLMManager,
        speaker_lock: threading.Lock,
        executor: futures.Executor | None = None,
    ):
        self._token = token
        self._llm_manager = llm_manager

        # Speaker lock is assumed to be provided by caller
        self._speaker_lock = speaker_lock

        # Speech recognition requires an API key in the environment variable
        os.environ["OPENAI_API_KEY"] = token

        if executor is None:
            self._executor = futures.ThreadPoolExecutor()
        else:
            self._executor = executor

    @staticmethod
    def _catch_all_executor(func: Callable, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(msg="Error in STTManager ", exc_info=e)

    def callback(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        # if TTS is speaking, skip recognition
        if self._speaker_lock.locked():
            logger.info("TTS is speaking, skipping STT recognition")
            return

        self._executor.submit(
            self._catch_all_executor, self.recognize, recognizer, audio
        )

    def stop(self, cancel_futures: bool = True):
        self._executor.shutdown(cancel_futures=cancel_futures)

    def recognize(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        logger.info("Detected audio for STT recognition")

        try:
            text: str = recognizer.recognize_openai(
                audio_data=audio,
                model="gpt-4o-mini-transcribe",
            )
        except Exception as e:
            logger.exception("STT recognition failed", exc_info=e)
            return

        text = text.lower().strip().removesuffix(".")
        if text is None or len(text) == 0:
            logger.info("No speech detected")
            return

        logger.info(f"Recognized speech: {text}")
        self._llm_manager.add_task(LLMTask(receiving_time=time.time(), text=text))
