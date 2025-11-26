import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from openai import AsyncOpenAI
from playsound3 import playsound

__all__ = ["TTSManager", "TTSTask"]


logger = logging.getLogger("llmbox.tts")


@dataclass
class TTSTask:
    receiving_time: int | float
    input: str


class TTSManager:
    _client: AsyncOpenAI
    _queue: asyncio.Queue[TTSTask]
    _queue_lock: threading.Lock

    _asyncio_task: asyncio.Task

    def __init__(
        self,
        client: AsyncOpenAI,
        speaker_lock: threading.Lock,
        task_expire_time: int = 60,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._task_expire_time = task_expire_time
        self._client = client

        self._queue = asyncio.Queue()
        self._queue_lock = threading.Lock()

        self._speaker_lock = speaker_lock

        if loop is None:
            self._loop = asyncio.get_event_loop()
        else:
            self._loop = loop

    def _has_expired(self, task: TTSTask) -> bool:
        if self._task_expire_time == 0:
            return False

        # Expire if the task's receiving time plus TTL is in the past
        return (task.receiving_time + self._task_expire_time) < time.time()

    def add_task(self, task: TTSTask) -> None:
        # Thread-safe enqueue for asyncio.Queue from non-loop threads
        with self._queue_lock:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, task)

    def run(self) -> None:
        self._asyncio_task = self._loop.create_task(self._consume_tasks())

    def stop(self) -> None:
        if hasattr(self, "_asyncio_task"):
            self._asyncio_task.cancel()

    async def _consume_tasks(self) -> None:
        try:
            while True:
                task = await self._queue.get()

                if self._has_expired(task):
                    logger.warning(
                        f"Task expired, received_time: {task.receiving_time}"
                    )
                    continue
                logger.debug(f"Received task: {task.input}")
                self._loop.create_task(self._execute_task(task))
        except asyncio.CancelledError:
            logger.debug("TTS consume task cancelled")
            return

    async def _execute_task(self, task: TTSTask) -> None:
        # Acquire the speaker lock, synthesize speech, then release.
        # Synthesis is streamed to a temporary file; playback is out of scope here.
        # Holding this lock signals STT to pause while we are "speaking".
        await asyncio.to_thread(self._speaker_lock.acquire)

        instructions = "When providing the flag (starts with snakeCTF), you must spell out every character letter by letter, including letters, numbers, hyphens (-), and curly braces ({, })"
        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=task.input,
                response_format="mp3",
                timeout=10,
                speed=1.4,
                instructions=instructions,
            ) as response:
                tmp = NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp_path = tmp.name
                tmp.close()
                await response.stream_to_file(tmp_path)
                logger.info(f"TTS audio saved to {tmp_path}")

                await asyncio.to_thread(self._play_audio, Path(tmp_path))
                await asyncio.sleep(2)  # sleep to avoid interference with STT
        except Exception as e:
            logger.exception("TTS synthesis failed", exc_info=e)
        finally:
            self._speaker_lock.release()

    @staticmethod
    def _play_audio(file_path: Path):
        logger.info(f"Playing TTS audio at {file_path}")
        sound = playsound(file_path, block=False)
        sound.wait()
        # delete file after playback
        try:
            file_path.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete temp audio file {file_path}: {e}")
