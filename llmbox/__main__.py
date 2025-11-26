import argparse
import asyncio
import logging
import threading

import httpx
import ollama
import sounddevice as sd  # type: ignore # noqa: F401
import speech_recognition as sr
from openai import AsyncOpenAI

from .llm_manager import LLMManager
from .stt_manager import STTManager
from .tts_manager import TTSManager

logger: logging.Logger


__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--token", required=True)
    parser.add_argument("-s", "--ollama-server", required=True)
    parser.add_argument("-u", "--ollama-username", required=True)
    parser.add_argument("-p", "--ollama-password", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument(
        "--no-auto-calibrate", dest="auto_calibrate", action="store_false", default=True
    )
    parser.add_argument(
        "--task-expire-time", type=int, default=120, help="Task expire time in seconds"
    )

    return parser


def build_logger(level: int) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)

    standard_logger = logging.getLogger("llmbox")
    return standard_logger


def main(
    token: str,
    host: str,
    username: str,
    password: str,
    auto_calibrate: bool,
    expire_time: int,
):
    logger.info("Setting up llmbox...")
    loop = asyncio.new_event_loop()

    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        logging.info(
            'Microphone with name "{1}" found for `Microphone(device_index={0})`'.format(
                index, name
            )
        )

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    openai_client = AsyncOpenAI(api_key=token, timeout=10, max_retries=3)
    ollama_client = ollama.AsyncClient(
        host=host,
        auth=httpx.BasicAuth(username=username, password=password),
    )

    speaker_lock = threading.Lock()
    tts_manager = TTSManager(
        task_expire_time=expire_time,
        client=openai_client,
        loop=loop,
        speaker_lock=speaker_lock,
    )
    llm_manager = LLMManager(
        client=ollama_client,
        task_expire_time=expire_time,
        tts_manager=tts_manager,
        loop=loop,
        flag="snakeCTF{llm_in_a_box_67}",
    )
    stt_manager = STTManager(
        token=token, llm_manager=llm_manager, speaker_lock=speaker_lock
    )

    # Calibration
    recognizer.energy_threshold = 4000
    recognizer.pause_threshold = 2.2
    recognizer.operation_timeout = 10
    recognizer.phrase_threshold = 0.5
    recognizer.non_speaking_duration = 0.8
    if auto_calibrate:
        logger.info("Calibrating microphone for ambient noise...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=5)

    logger.info("Initializing...")
    stopper = recognizer.listen_in_background(
        microphone, callback=stt_manager.callback, phrase_time_limit=30
    )
    tts_manager.run()
    llm_manager.run()

    logger.info("llmbox is running. Press Ctrl+C to stop.")

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    # Safely stop the background listener
    stopper()
    stt_manager.stop()
    llm_manager.stop()
    tts_manager.stop()

    # Stop the asyncio loop thread and close the loop
    loop.stop()
    loop.close()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logger = build_logger(
        logging.DEBUG if args.verbose else logging.INFO,
    )

    main(
        token=args.token,
        username=args.ollama_username,
        password=args.ollama_password,
        host=args.ollama_server,
        auto_calibrate=args.auto_calibrate,
        expire_time=args.task_expire_time,
    )
