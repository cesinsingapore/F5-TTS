import struct
import time
import logging
import asyncio
import numpy as np
import pyaudio
from kafka import KafkaConsumer
from queue import Queue

from cls import trigger_motions

# ✅ Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Kafka Configuration
KAFKA_BROKER = "localhost:29092"
KAFKA_TOPIC_AUDIO = "audio_chunks"  # Same as original server

# ✅ Audio Configuration
SAMPLE_RATE = 24000
BUFFER_SIZE = 2048

# ✅ Motion Queue to hold incoming motion data
motion_queue = Queue()

async def play_audio_stream():
    """ Continuously listens to Kafka topic and plays audio in real-time. """
    while True:
        try:
            buffer = b''
            first_chunk_time = None

            # ✅ Initialize audio player
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=SAMPLE_RATE,
                            output=True,
                            frames_per_buffer=2048)

            # ✅ Set up Kafka Consumer
            consumer = KafkaConsumer(
                KAFKA_TOPIC_AUDIO,
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda v: struct.unpack(f'{len(v) // 4}f', v),
                enable_auto_commit=True,
                auto_offset_reset='latest',
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                max_poll_interval_ms=600000
            )

            total_chunks = 0
            total_size = 0

            logger.info(f"🎧 Listening for audio chunks on topic: {KAFKA_TOPIC_AUDIO}...")

            for message in consumer:
                audio_chunk = message.value
                if not audio_chunk:
                    logger.info("🛑 Received end-of-audio signal. Waiting for next stream...")
                    buffer = b''
                    break

                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.info("✅ First chunk received, starting playback.")


                total_chunks += 1
                chunk_size = len(audio_chunk)
                total_size += chunk_size
                logger.info(f"🔊 Received audio chunk of size {chunk_size}")

                buffer += struct.pack(f'{chunk_size}f', *audio_chunk)
                while len(buffer) >= BUFFER_SIZE:
                    audio_array = np.frombuffer(buffer[:BUFFER_SIZE], dtype=np.float32)
                    stream.write(audio_array.tobytes())
                    buffer = buffer[BUFFER_SIZE:]

            if buffer:
                logger.info(f"🎵 Writing remaining buffer of size {len(buffer)}")
                audio_array = np.frombuffer(buffer, dtype=np.float32)
                stream.write(audio_array.tobytes())

            await asyncio.sleep(0.5)

            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info(f"✅ Audio playback finished. Total chunks: {total_chunks}, Total size: {total_size} floats")

        except Exception as e:
            logger.error(f"❌ Kafka connection error: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(play_audio_stream())
