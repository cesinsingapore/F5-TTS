import os
import socket
import struct
import tempfile
from importlib.resources import files

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
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.output_file = output_file
        self.sampling_rate = sampling_rate
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audio_data = []
        self.file_opened = False

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
        self.join()
        if self.file_opened:
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sampling_rate)
                for chunk in self.audio_data:
                    if chunk is not None:
                        wf.writeframes(chunk.tobytes())


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
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: struct.pack(f'{len(v)}f', *v)
        )
        self.sampling_rate = 24000
        self.file_writer_thread = None
        self.update_reference(ref_audio, ref_text)

    def update_reference(self, ref_audio, ref_text):
        self.ref_audio = ref_audio
        self.ref_text = ref_text

    def generate_stream(self, text, character_name, output_file="output_chinese.wav"):
        logger.info("Generating audio with Chinese F5TTS...")

        # ✅ Pisahkan motion
        if "|||" in text:
            text, motion_raw = text.split("|||", 1)
            motions = [m.strip() for m in motion_raw.split(",") if m.strip()]
        else:
            motions = []

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

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) == 0:
                continue

            chunk_duration = len(chunk) / self.sampling_rate
            total_duration += chunk_duration

            self.file_writer_thread.add_chunk(chunk)
            self.producer.send(self.kafka_topic, value=chunk)

            total_chunks += 1
            total_size += len(chunk)
            logger.info(f"Writing chunk to file: mean={np.mean(chunk)}, std={np.std(chunk)}")

        logger.info("✅ Finished sending all Chinese audio chunks to Kafka")
        self.producer.send(self.kafka_topic, value=[])  # Signal end-of-audio
        self.file_writer_thread.stop()

        # ✅ Send motion timeline if present
        if motions:
            interval = total_duration / len(motions) if motions else 0
            motion_timeline = [
                {"name": m, "start": round(i * interval, 2)}
                for i, m in enumerate(motions)
            ]

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
        self.producer = KafkaProducer(bootstrap_servers=kafka_servers,
                                      value_serializer=lambda v: struct.pack(f'{len(v)}f', *v))
        self.model = self.load_ema_model(ckpt_file, vocab_file, dtype)
        self.vocoder = self.load_vocoder_model()

        self.update_reference(ref_audio, ref_text)
        self._warm_up()
        self.file_writer_thread = None
        self.first_package = True

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
            text, motion_raw = text.split("|||", 1)
            motions = [m.strip() for m in motion_raw.split(",") if m.strip()]
        else:
            motions = []

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

        previous_chunk = None
        cross_fade_samples = int(cross_fade_duration * self.sampling_rate)

        if self.file_writer_thread is not None:
            self.file_writer_thread.stop()
        self.file_writer_thread = AudioFileWriterThread(output_file, self.sampling_rate)
        self.file_writer_thread.start()

        for item in audio_stream:
            audio_chunk, final_sample_rate = item

            if not generation_started:
                generation_time = time.time() - start_time
                logger.info(f"Audio generation started. Initial processing took {generation_time:.2f} seconds.")
                generation_started = True

            if len(audio_chunk) > 0:
                chunk_duration = len(audio_chunk) / final_sample_rate
                total_duration += chunk_duration

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
                logger.info(f"Producing audio chunk of size {chunk_size} to Kafka")
                self.producer.send(self.kafka_topic, value=audio_chunk)

                logger.info(f"Writing chunk to file: mean={np.mean(audio_chunk)}, std={np.std(audio_chunk)}")
                self.file_writer_thread.add_chunk(audio_chunk)

        logger.info(f"Total chunks produced: {total_chunks}, Total size: {total_size} floats")

        end_time = time.time()
        generation_time = end_time - start_time
        audio_duration = total_size / self.sampling_rate if self.sampling_rate > 0 else 0
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        logger.info(f"Real-Time Factor (RTF): {rtf:.2f}")

        logger.info("Producing end-of-audio signal to Kafka")
        self.producer.send(self.kafka_topic, value=[])

        self.file_writer_thread.stop()

        # ✅ Kirim motion timeline jika ada motion
        # ✅ Send motion timeline if present
        if motions:
            interval = total_duration / len(motions) if motions else 0
            motion_timeline = [
                {"name": m, "start": round(i * interval, 2)}
                for i, m in enumerate(motions)
            ]

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
                "session_id": session_id,
                "character_id": character_id,
                "character_name": character_name_from_file
            })
            logger.info(f"📤 Motion timeline sent to Kafka: {motion_timeline} with session_id: {session_id}")
        logger.info("✅ Finished sending all English audio chunks to Kafka")


def handle_client(conn, addr, processors):
    try:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                data = conn.recv(1024)
                if not data:
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
                        lang = "zh"  # default fallback

                    # ✅ Parse text ||| motion1,motion2 | filename.wav
                    if "|||" in text_with_motion:
                        text_part, motion_and_filename = text_with_motion.split("|||", 1)
                        if "|" in motion_and_filename:
                            motion_raw, output_filename = motion_and_filename.split("|", 1)
                        else:
                            motion_raw = motion_and_filename
                            output_filename = f"{character_name}_{int(time.time())}.wav"
                    else:
                        text_part = text_with_motion
                        motion_raw = ""
                        output_filename = f"{character_name}_{int(time.time())}.wav"

                    motions = [m.strip() for m in motion_raw.split(",") if m.strip()]
                    full_text = f"{text_part}|||{','.join(motions)}" if motions else text_part

                    logger.info(f"🗣️ Lang: {lang}, Character: {character_name}, Text: {text_part}, Motions: {motions}, Output: {output_filename}")

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
                processor.generate_stream(full_text, character_name, output_file=output_filename)

    except Exception as e:
        logger.error(f"Error handling client: {e}")
        traceback.print_exc()



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

def start_server(host, port, processor):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        logger.info(f"Server started on {host}:{port}")
        while True:
            conn, addr = s.accept()
            logger.info(f"Connected by {addr}")
            handle_client(conn, addr, processor)

if __name__ == "__main__":
    try:
        processors = {
            "en": TTSStreamingProcessor(
                ckpt_file="ckpts/mix-singlish/model_258000.pt",
                vocab_file="",
                ref_audio="./tests/ref_audio/ref2.mp3",
                ref_text="I'm not sure, you may want to check with the security of ion orchard",
                kafka_topic="audio_chunks",
                kafka_servers=["192.168.194.42:29092"]
            ),
            "zh": ChineseTTSProcessor(
                ckpt_file="ckpts/chinese/model_1200000.pt",
                vocab_file="./data/Emilia_ZH_EN_pinyin/vocab.txt",
                ref_audio="./tests/ref_audio/ref_ch.mp3",
                ref_text="我不确定，你可能需要向 Ion Orchard 的保安部门查询一下。",
                kafka_topic="audio_chunks_zh",
                kafka_servers=["192.168.194.42:29092"]
            )
        }

        host = '0.0.0.0'
        port = 9998
        start_server(host, port, processors)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()
