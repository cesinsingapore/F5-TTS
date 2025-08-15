import os
import socket
import struct
import tempfile
from importlib.resources import files
import json
import hashlib
import signal
import sys
import threading

import torch
import torchaudio
import logging
import wave
import numpy as np
from cached_path import cached_path
from hydra.utils import get_class
from kafka import KafkaProducer
from nltk.tokenize import sent_tokenize
from omegaconf import OmegaConf

from f5_tts.api import F5TTS
from infer.utils_infer import preprocess_ref_audio_text, load_vocoder, load_model, infer_batch_process
from model.backbones.dit import DiT
from pymongo import MongoClient
from bson import ObjectId
import time
import traceback
import queue

# Configure logging
import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging with both file and console output
log_filename = f"logs/f5tts_server_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"📝 Logs will be saved to: {log_filename}")

# Global shutdown flag
shutdown_flag = threading.Event()
active_threads = []
server_socket = None

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "simulation_db"

# Initialize MongoDB client and collection reference
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
characters_collection = db["characters"]

class AudioFileWriterThread(threading.Thread):
    def __init__(self, output_file, sampling_rate):
        super().__init__()
        self.daemon = True  # Dies when main thread dies
        self.output_file = output_file
        self.sampling_rate = sampling_rate
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audio_data = []
        self.file_opened = False
        active_threads.append(self)

    def clear_queue(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                continue

    def clear_audio_data(self):
        self.audio_data = []

    def run(self):
        logger.info("AudioFileWriterThread started.")
        self.clear_queue()

        with wave.open(self.output_file, 'wb') as wf:
            self.file_opened = True
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sampling_rate)

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    chunk = self.queue.get(timeout=0.1)
                    if chunk is not None:
                        chunk = np.int16(chunk * 32767)
                        self.audio_data.append(chunk)
                        logger.info(f"Writing chunk to file: mean={np.mean(chunk)}, std={np.std(chunk)}")
                        wf.writeframes(chunk.tobytes())
                except queue.Empty:
                    continue

    def add_chunk(self, chunk):
        self.queue.put(chunk)

    def stop(self):
        self.stop_event.set()
        try:
            self.join(timeout=2.0)  # Wait max 2 seconds
        except:
            pass
        if self.file_opened:
            try:
                with wave.open(self.output_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sampling_rate)
                    for chunk in self.audio_data:
                        if chunk is not None:
                            wf.writeframes(chunk.tobytes())
            except Exception as e:
                logger.error(f"Failed to save final audio file: {e}")
        if self in active_threads:
            active_threads.remove(self)


class ChineseTTSProcessor:
    def __init__(self, ckpt_file, vocab_file, ref_audio, ref_text, kafka_topic, kafka_servers, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = F5TTS(
            model="E2TTS_Base",
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            device=self.device,
            use_ema=True
        )
        self.kafka_topic = kafka_topic
        
        # Optimized Kafka producer for low-latency streaming
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: struct.pack(f'{len(v)}f', *v),
            # Low-latency optimizations
            batch_size=1,  # Send immediately, don't batch
            linger_ms=0,   # Don't wait to batch
            acks=1,        # Only wait for leader acknowledgment
            compression_type=None,  # No compression for speed
            max_in_flight_requests_per_connection=10,
            buffer_memory=33554432,  # 32MB buffer
            retries=3
        )
        
        # Debug producer for metadata logging
        self.debug_producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            batch_size=1,
            linger_ms=0,
            acks=1
        )
        
        self.sampling_rate = 24000
        self.file_writer_thread = None
        self.chunk_sequence = 0
        self.session_id = None
        self.update_reference(ref_audio, ref_text)
        
        # Keep track of producers for cleanup
        active_threads.extend([self.producer, self.debug_producer])

    def update_reference(self, ref_audio, ref_text):
        self.ref_audio = ref_audio
        self.ref_text = ref_text

    def generate_stream(self, text, character_name, output_file="output_chinese.wav"):
        logger.info("Generating audio with Chinese F5TTS...")

        # ✅ Pisahkan motion
        if "|||" in text:
            parts = text.split("|||")
            text = parts[0].strip()
            motions = [m.strip() for m in parts[1].split(",") if m.strip()] if len(parts) > 1 else []
            facial_expression = parts[2].strip() if len(parts) > 2 else "normal"
        else:
            motions = []
            facial_expression = "normal"

        # ✅ Log informasi
        logger.info(f"🗣️ Character: {character_name}, Facial Expression: {facial_expression}")
        logger.info(f"🗣️ Text: {text}")
        logger.info(f"🎭 Motions: {motions}")

        # ✅ Inference sekali saja lalu load waveform
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.infer(
                gen_text=text.strip().lower(),
                ref_text=self.ref_text.strip().lower(),
                ref_file=self.ref_audio,
                file_wave=f.name,
                speed=0.3 if len(text) < 10 else 0.7,
                seed=42,
                remove_silence=False
            )
            waveform, sr = torchaudio.load(f.name)
            audio_data = waveform[0].numpy()

        # ✅ Stop previous writer
        if self.file_writer_thread is not None:
            self.file_writer_thread.stop()

        # ✅ Start writer thread
        self.file_writer_thread = AudioFileWriterThread(output_file, self.sampling_rate)
        self.file_writer_thread.start()

        chunk_size = 2048
        total_chunks = 0
        total_size = 0
        total_duration = 0.0  # ⏱️ Track durasi semua chunk
        self.chunk_sequence = 0
        self.session_id = f"{character_name}_{int(time.time())}"
        stream_start_time = time.time()

        logger.info(f"🚀 [STREAM_START] Session: {self.session_id}, Total audio length: {len(audio_data)} samples")

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) == 0:
                continue

            chunk_duration = len(chunk) / self.sampling_rate
            total_duration += chunk_duration
            chunk_timestamp = time.time()
            
            # Create chunk hash for verification
            chunk_bytes = chunk.tobytes()
            chunk_hash = hashlib.md5(chunk_bytes).hexdigest()[:8]
            
            # Send audio chunk to Kafka
            self.file_writer_thread.add_chunk(chunk)
            future = self.producer.send(self.kafka_topic, value=chunk)
            
            # Log detailed chunk info for debugging
            chunk_info = {
                "session_id": self.session_id,
                "sequence": self.chunk_sequence,
                "timestamp": chunk_timestamp,
                "offset_ms": round((chunk_timestamp - stream_start_time) * 1000, 2),
                "chunk_size": len(chunk),
                "chunk_duration_ms": round(chunk_duration * 1000, 2),
                "chunk_hash": chunk_hash,
                "topic": self.kafka_topic,
                "mean": float(np.mean(chunk)),
                "std": float(np.std(chunk)),
                "language": "zh",
                "character": character_name
            }
            
            # Send debug info to separate topic
            self.debug_producer.send(f"{self.kafka_topic}_debug", value=chunk_info)
            
            total_chunks += 1
            total_size += len(chunk)
            self.chunk_sequence += 1
            
            logger.info(f"📤 [CHUNK_SENT] Seq:{self.chunk_sequence}, Size:{len(chunk)}, Hash:{chunk_hash}, Offset:{chunk_info['offset_ms']}ms")
            
            # Check if send was successful
            try:
                record_metadata = future.get(timeout=1)
                logger.debug(f"✅ Chunk {self.chunk_sequence} delivered to partition {record_metadata.partition} at offset {record_metadata.offset}")
            except Exception as e:
                logger.error(f"❌ Failed to send chunk {self.chunk_sequence}: {e}")

        # Send end-of-stream marker
        end_timestamp = time.time()
        end_info = {
            "session_id": self.session_id,
            "sequence": self.chunk_sequence + 1,
            "timestamp": end_timestamp,
            "offset_ms": round((end_timestamp - stream_start_time) * 1000, 2),
            "chunk_size": 0,
            "chunk_duration_ms": 0,
            "chunk_hash": "END_OF_STREAM",
            "topic": self.kafka_topic,
            "total_chunks": total_chunks,
            "total_duration_ms": round(total_duration * 1000, 2),
            "language": "zh",
            "character": character_name
        }
        
        # Send end signals with timeout
        try:
            self.producer.send(self.kafka_topic, value=[]).get(timeout=1)
            self.debug_producer.send(f"{self.kafka_topic}_debug", value=end_info).get(timeout=1)
        except Exception as e:
            logger.warning(f"Failed to send end signals: {e}")
        
        logger.info(f"✅ [STREAM_END] Session: {self.session_id}, Total chunks: {total_chunks}, Duration: {round(total_duration, 2)}s")
        if self.file_writer_thread:
            self.file_writer_thread.stop()

        # ✅ Send motion timeline if present
        if motions:
            interval = total_duration / len(motions) if motions else 0
            motion_timeline = []
            for i, m in enumerate(motions):
                start_time = round(i * interval, 2)
                end_time = round((i + 1) * interval, 2) if i < len(motions) - 1 else round(total_duration, 2)
                motion_timeline.append({
                    "name": m,
                    "start": start_time,
                    "end": end_time
                })

            from kafka import KafkaProducer
            import json
            motion_producer = KafkaProducer(
                bootstrap_servers=["192.168.194.42:29092"],
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            filename = os.path.basename(output_file)
            try:
                parts = filename.replace(".wav", "").split("_")
                if len(parts) >= 3:
                    character_name_from_file = parts[0]
                    session_id = parts[1]
                    character_id = parts[2]
                else:
                    session_id = f"{character_name}_{int(time.time())}"  # fallback
                    character_id = "unknown"
                    character_name_from_file = character_name
                    logger.warning("⚠️ Failed to parse session_id and character_id from filename.")
            except Exception as e:
                session_id = f"{character_name}_{int(time.time())}"  # fallback
                character_id = "unknown"
                character_name_from_file = character_name
                logger.error(f"❌ Error parsing output filename: {e}")

            motion_producer.send("audio_motion_meta", value={
                "motions": motion_timeline,
                "facial_expression": facial_expression,
                "session_id": session_id,
                "character_id": character_id,
                "character_name": character_name_from_file
            })
            logger.info(f"📤 Motion timeline sent to Kafka: {motion_timeline} with session_id: {session_id}")

class TTSStreamingProcessor:
    def __init__(self, ckpt_file, vocab_file, ref_audio, ref_text, kafka_topic, kafka_servers, device=None, dtype=torch.float32):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/F5TTS_Base.yaml")))
        self.model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self.model_arc = model_cfg.model.arch
        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.sampling_rate = model_cfg.model.mel_spec.target_sample_rate
        self.kafka_topic = kafka_topic
        
        # Optimized Kafka producer for low-latency streaming
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: struct.pack(f'{len(v)}f', *v),
            # Low-latency optimizations
            batch_size=1,  # Send immediately, don't batch
            linger_ms=0,   # Don't wait to batch
            acks=1,        # Only wait for leader acknowledgment
            compression_type=None,  # No compression for speed
            max_in_flight_requests_per_connection=10,
            buffer_memory=33554432,  # 32MB buffer
            retries=3
        )
        
        # Debug producer for metadata logging
        self.debug_producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            batch_size=1,
            linger_ms=0,
            acks=1
        )
        
        self.model = self.load_ema_model(ckpt_file, vocab_file, dtype)
        self.vocoder = self.load_vocoder_model()

        self.update_reference(ref_audio, ref_text)
        self._warm_up()
        self.file_writer_thread = None
        self.first_package = True
        self.chunk_sequence = 0
        self.session_id = None
        
        # Keep track of producers for cleanup
        active_threads.extend([self.producer, self.debug_producer])

    def load_ema_model(self, ckpt_file, vocab_file, dtype):
        return load_model(
            self.model_cls,
            self.model_arc,
            ckpt_path=ckpt_file,
            mel_spec_type=self.mel_spec_type,
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

    def load_vocoder_model(self):
        return load_vocoder(vocoder_name=self.mel_spec_type, is_local=False, local_path=None, device=self.device)

    def update_reference(self, ref_audio, ref_text):
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
        self.audio, self.sr = torchaudio.load(self.ref_audio)

        ref_audio_duration = self.audio.shape[-1] / self.sr
        ref_text_byte_len = len(self.ref_text.encode("utf-8"))
        self.max_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration))
        self.few_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 2)
        self.min_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 4)

    def _warm_up(self):
        logger.info("Warming up the model...")
        gen_text = "Warm-up text for the model."
        for _ in infer_batch_process(
            (self.audio, self.sr),
            self.ref_text,
            [gen_text],
            self.model,
            self.vocoder,
            progress=None,
            device=self.device,
            streaming=True,
        ):
            pass
        logger.info("Warm-up completed.")

    def split_text_into_batches(self, text, max_batch_length=20):
        sentences = sent_tokenize(text)
        batches = []
        current_batch = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_batch_length:
                if current_batch:
                    batches.append(' '.join(current_batch))
                current_batch = [sentence]
                current_length = sentence_length
            else:
                current_batch.append(sentence)
                current_length += sentence_length
        if current_batch:
            batches.append(' '.join(current_batch))
        return batches

    def generate_stream(self, text, character_name, output_file="output_english.wav", cross_fade_duration=0.20):
        if "|||" in text:
            parts = text.split("|||")
            text = parts[0].strip()
            motions = [m.strip() for m in parts[1].split(",") if m.strip()] if len(parts) > 1 else []
            if len(parts) > 2:
                fe_and_filename = parts[2].strip().split("|", 1)
                facial_expression = fe_and_filename[0].strip()
                output_filename = fe_and_filename[1].strip() if len(
                    fe_and_filename) > 1 else f"{character_name}_{int(time.time())}.wav"
            else:
                facial_expression = "normal"
                output_filename = f"{character_name}_{int(time.time())}.wav"
        else:
            motions = []
            facial_expression = "normal"

        # ✅ Log informasi
        logger.info(f"🗣️ Character: {character_name}, Facial Expression: {facial_expression}")
        logger.info(f"🗣️ Text: {text}")
        logger.info(f"🎭 Motions: {motions}")
        logger.info("Generating audio with English F5TTS...")

        start_time = time.time()
        speed = 0.3 if len(text) < 10 else 1
        text_batches = self.split_text_into_batches(text)
        logger.info(f"Text batches: {text_batches}")
        audio_stream = infer_batch_process(
            (self.audio, self.sr),
            self.ref_text,
            text_batches,
            self.model,
            self.vocoder,
            device=self.device,
            streaming=True,
            chunk_size=2048,
            speed=speed
        )

        generation_started = False
        total_chunks = 0
        total_size = 0
        total_duration = 0.0
        self.chunk_sequence = 0
        self.session_id = f"{character_name}_{int(time.time())}"
        stream_start_time = time.time()

        previous_chunk = None
        cross_fade_samples = int(cross_fade_duration * self.sampling_rate)

        if self.file_writer_thread is not None:
            self.file_writer_thread.stop()
        self.file_writer_thread = AudioFileWriterThread(output_filename, self.sampling_rate)
        self.file_writer_thread.start()
        
        logger.info(f"🚀 [STREAM_START] Session: {self.session_id}, Text batches: {len(text_batches)}")

        for item in audio_stream:
            audio_chunk, final_sample_rate = item

            if not generation_started:
                generation_time = time.time() - start_time
                logger.info(f"Audio generation started. Initial processing took {generation_time:.2f} seconds.")
                generation_started = True

            if len(audio_chunk) > 0:
                chunk_duration = len(audio_chunk) / final_sample_rate
                total_duration += chunk_duration
                chunk_timestamp = time.time()

                logger.info(f"Generated audio chunk of size: {len(audio_chunk)}")
                if previous_chunk is not None:
                    if len(previous_chunk) >= cross_fade_samples and len(audio_chunk) >= cross_fade_samples:
                        fade_in = np.linspace(0, 1, cross_fade_samples)
                        fade_out = 1 - fade_in

                        prev_overlap = previous_chunk[-cross_fade_samples:]
                        next_overlap = audio_chunk[:cross_fade_samples]
                        overlap = prev_overlap * fade_out + next_overlap * fade_in
                        audio_chunk[:cross_fade_samples] = overlap

                previous_chunk = audio_chunk
                chunk_size = len(audio_chunk)
                total_chunks += 1
                total_size += chunk_size
                
                # Create chunk hash for verification
                chunk_bytes = audio_chunk.tobytes()
                chunk_hash = hashlib.md5(chunk_bytes).hexdigest()[:8]
                
                # Log detailed chunk info for debugging
                chunk_info = {
                    "session_id": self.session_id,
                    "sequence": self.chunk_sequence,
                    "timestamp": chunk_timestamp,
                    "offset_ms": round((chunk_timestamp - stream_start_time) * 1000, 2),
                    "chunk_size": chunk_size,
                    "chunk_duration_ms": round(chunk_duration * 1000, 2),
                    "chunk_hash": chunk_hash,
                    "topic": self.kafka_topic,
                    "mean": float(np.mean(audio_chunk)),
                    "std": float(np.std(audio_chunk)),
                    "language": "en",
                    "character": character_name,
                    "cross_faded": previous_chunk is not None
                }
                
                # Send debug info to separate topic
                self.debug_producer.send(f"{self.kafka_topic}_debug", value=chunk_info)
                
                logger.info(f"📤 [CHUNK_SENT] Seq:{self.chunk_sequence}, Size:{chunk_size}, Hash:{chunk_hash}, Offset:{chunk_info['offset_ms']}ms")
                
                # Send audio chunk to Kafka
                future = self.producer.send(self.kafka_topic, value=audio_chunk)
                
                # Check if send was successful
                try:
                    record_metadata = future.get(timeout=1)
                    logger.debug(f"✅ Chunk {self.chunk_sequence} delivered to partition {record_metadata.partition} at offset {record_metadata.offset}")
                except Exception as e:
                    logger.error(f"❌ Failed to send chunk {self.chunk_sequence}: {e}")
                
                self.file_writer_thread.add_chunk(audio_chunk)
                self.chunk_sequence += 1

        # Calculate timing metrics first
        end_time = time.time()
        generation_time = end_time - start_time
        audio_duration = total_size / self.sampling_rate if self.sampling_rate > 0 else 0
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        
        # Send end-of-stream marker
        end_timestamp = time.time()
        end_info = {
            "session_id": self.session_id,
            "sequence": self.chunk_sequence + 1,
            "timestamp": end_timestamp,
            "offset_ms": round((end_timestamp - stream_start_time) * 1000, 2),
            "chunk_size": 0,
            "chunk_duration_ms": 0,
            "chunk_hash": "END_OF_STREAM",
            "topic": self.kafka_topic,
            "total_chunks": total_chunks,
            "total_duration_ms": round(total_duration * 1000, 2),
            "language": "en",
            "character": character_name,
            "rtf": rtf
        }
        
        logger.info(f"Total chunks produced: {total_chunks}, Total size: {total_size} floats")
        logger.info(f"Real-Time Factor (RTF): {rtf:.2f}")
        
        # Send end signals with timeout
        try:
            self.producer.send(self.kafka_topic, value=[]).get(timeout=1)
            self.debug_producer.send(f"{self.kafka_topic}_debug", value=end_info).get(timeout=1)
        except Exception as e:
            logger.warning(f"Failed to send end signals: {e}")
        
        logger.info(f"✅ [STREAM_END] Session: {self.session_id}, Total chunks: {total_chunks}, Duration: {round(total_duration, 2)}s, RTF: {rtf:.2f}")
        
        if self.file_writer_thread:
            self.file_writer_thread.stop()

        # ✅ Kirim motion timeline jika ada motion
        # ✅ Send motion timeline if present
        if motions:
            interval = total_duration / len(motions) if motions else 0
            motion_timeline = []
            for i, m in enumerate(motions):
                start_time = round(i * interval, 2)
                end_time = round((i + 1) * interval, 2) if i < len(motions) - 1 else round(total_duration, 2)
                motion_timeline.append({
                    "name": m,
                    "start": start_time,
                    "end": end_time
                })

            from kafka import KafkaProducer
            import json
            motion_producer = KafkaProducer(
                bootstrap_servers=["localhost:29092"],
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            filename = os.path.basename(output_file)
            try:
                parts = fe_and_filename[1].replace(".wav", "").split("_")
                if len(parts) >= 3:
                    character_name_from_file = parts[0]
                    session_id = parts[1]
                    character_id = parts[2]
                else:
                    session_id = f"{character_name}_{int(time.time())}"  # fallback
                    character_id = "unknown"
                    character_name_from_file = character_name
                    logger.warning("⚠️ Failed to parse session_id and character_id from filename.")
            except Exception as e:
                session_id = f"{character_name}_{int(time.time())}"  # fallback
                character_id = "unknown"
                character_name_from_file = character_name
                logger.error(f"❌ Error parsing output filename: {e}")

            motion_producer.send("audio_motion_meta", value={
                "motions": motion_timeline,
                "facial_expression": facial_expression,
                "session_id": session_id,
                "character_id": character_id,
                "character_name": character_name_from_file
            })
            logger.info(f"📤 Motion timeline sent to Kafka: {motion_timeline} with session_id: {session_id}")
        logger.info("✅ Finished sending all English audio chunks to Kafka")


def handle_client(conn, addr, processors):
    client_thread = threading.current_thread()
    # Note: Cannot set daemon on already running thread
    active_threads.append(client_thread)
    
    try:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(1.0)  # 1 second timeout for recv operations
            
            while not shutdown_flag.is_set():
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                except socket.timeout:
                    continue  # Check shutdown flag and continue
                except Exception as e:
                    logger.error(f"Socket error: {e}")
                    break
                data_str = data.decode("utf-8").strip()
                parts = data_str.split('|', 2)

                try:
                    lang, character_name, text_with_motion = parts
                    lang = lang.lower()
                    if "en" in lang:
                        lang = "en"
                    elif "zh" in lang:
                        lang = "zh"
                    else:
                        lang = "en"  # fallback

                    # ✅ Parse text ||| motions ||| facial ||| filename |||
                    if "|||" in text_with_motion:
                        parts_ = text_with_motion.split("|||")
                        text_part = parts_[0].strip()
                        motion_raw = parts_[1].strip() if len(parts_) > 1 else ""
                        facial_expression = parts_[2].strip() if len(parts_) > 2 else "normal"
                        output_filename = parts_[3].strip() if len(parts_) > 3 else f"{character_name}_{int(time.time())}.wav"
                    else:
                        text_part = text_with_motion
                        motion_raw = ""
                        facial_expression = "normal"
                        output_filename = f"{character_name}_{int(time.time())}.wav"

                    motions = [m.strip() for m in motion_raw.split(",") if m.strip()]
                    full_text = f"{text_part}|||{','.join(motions)}|||{facial_expression}"

                    logger.info(f"🗣️ Lang: {lang}, Character: {character_name}, Text: {text_part}, Motions: {motions}, Facial: {facial_expression}, Output: {output_filename}")

                except ValueError:
                    logger.error("Invalid format. Use: lang|character|text")
                    break

                processor = processors.get(lang)
                if not processor:
                    logger.error(f"Unsupported language: {lang}")
                    break

                ref_audio, ref_text = get_ref_audio_text(character_name)
                if not ref_audio or not ref_text:
                    logger.error(f"Missing reference data for character '{character_name}'")
                    break

                processor.update_reference(ref_audio, ref_text)
                if not shutdown_flag.is_set():
                    processor.generate_stream(full_text, character_name, output_file=output_filename)

    except Exception as e:
        logger.error(f"Error handling client: {e}")
        traceback.print_exc()
    finally:
        if client_thread in active_threads:
            active_threads.remove(client_thread)
        logger.info(f"Client {addr} disconnected")



def get_ref_audio_text(character_name):
    """Retrieve reference audio and text from MongoDB for a given character."""
    try:
        character_data = characters_collection.find_one({"name": character_name}, {"voice_path": 1, "ref_text": 1})

        if not character_data:
            logger.warning(f"⚠️ Character '{character_name}' not found in database.")
            return None, None  # Handle missing character data

        ref_audio = character_data.get("voice_path", "").strip()
        ref_text = character_data.get("ref_text", "").strip()

        if not ref_audio or not ref_text:
            logger.warning(f"⚠️ Missing ref_audio or ref_text for character '{character_name}'")
            return None, None  # Handle missing fields

        logger.info(f"✅ Retrieved ref_audio: {ref_audio}, ref_text: {ref_text} for {character_name}")
        return ref_audio, ref_text

    except Exception as e:
        logger.error(f"❌ Error retrieving reference data for '{character_name}': {e}")
        return None, None  # Handle database errors gracefully

def start_server(host, port, processors):
    global server_socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.settimeout(1.0)  # 1 second timeout for accept
    
    try:
        # Try binding to the port
        try:
            server_socket.bind((host, port))
        except OSError as e:
            if e.winerror == 10013:  # Windows permission denied
                logger.error(f"❌ Permission denied on port {port}. Try running as Administrator or use a different port (>1024)")
                logger.info(f"💡 Suggestion: Try port 8888 or 9999 instead of {port}")
                return
            else:
                raise e
                
        server_socket.listen(5)
        logger.info(f"🚀 Server started on {host}:{port} - Press Ctrl+C to stop")
        
        while not shutdown_flag.is_set():
            try:
                conn, addr = server_socket.accept()
                logger.info(f"🔗 Connected by {addr}")
                
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=handle_client, 
                    args=(conn, addr, processors),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                continue  # Check shutdown flag and continue
            except Exception as e:
                if not shutdown_flag.is_set():
                    logger.error(f"Accept error: {e}")
                break
                
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup_server()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    shutdown_flag.set()
    
def cleanup_server():
    """Clean up server resources."""
    global server_socket
    
    logger.info("🧹 Cleaning up server resources...")
    
    # Close server socket
    if server_socket:
        try:
            server_socket.close()
            logger.info("✅ Server socket closed")
        except Exception as e:
            logger.error(f"Error closing server socket: {e}")
    
    # Close Kafka producers and wait for threads to finish
    kafka_producers = [item for item in active_threads if hasattr(item, 'close')]
    threads = [item for item in active_threads if hasattr(item, 'is_alive')]
    
    logger.info(f"⏳ Cleaning up {len(kafka_producers)} Kafka producers and {len(threads)} threads...")
    
    # Close Kafka producers first
    for producer in kafka_producers:
        try:
            producer.close(timeout=2)
            logger.debug("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")
    
    # Wait for threads to finish
    for thread in threads:
        try:
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not finish in time")
        except Exception as e:
            logger.error(f"Error joining thread: {e}")
    
    # Close MongoDB connection
    try:
        client.close()
        logger.info("✅ MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB: {e}")
    
    logger.info("🏁 Server shutdown complete")

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
        processors = {
            "en": TTSStreamingProcessor(
                ckpt_file="ckpts/mix-singlish/model_258000.pth",
                vocab_file="ckpts/mix-singlish/vocab.txt",
                ref_audio="./tests/ref_audio/ref2.mp3",
                ref_text="I'm not sure, you may want to check with the security of ion orchard",
                kafka_topic="audio_chunks",
                kafka_servers=["192.168.1.69:29092"]
            ),
            "zh": ChineseTTSProcessor(
                ckpt_file="ckpts/chinese/model_1200000.pt",
                vocab_file="./data/Emilia_ZH_EN_pinyin/vocab.txt",
                ref_audio="./tests/ref_audio/ref_ch.mp3",
                ref_text="我不确定，你可能需要向 Ion Orchard 的保安部门查询一下。",
                kafka_topic="audio_chunks_zh",
                kafka_servers=["192.168.1.69:29092"]
            )
        }

        host = '0.0.0.0'
        port = 9998
        logger.info(f"🚀 Starting F5-TTS Socket Server on port {port}")
        start_server(host, port, processors)

    except KeyboardInterrupt:
        logger.info("\n🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        shutdown_flag.set()
        cleanup_server()
