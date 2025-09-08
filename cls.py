# motion_listener.py

import json
import logging
import time
import threading
from kafka import KafkaConsumer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka Config
KAFKA_BROKER = "localhost:29092"
KAFKA_TOPIC_MOTION = "audio_motion_realtime"  # Same as original server

# Optional: fine-tune motion offset
MOTION_COMPENSATION_OFFSET = -0.2

def trigger_motions(motion_data):
    # Handle new real-time motion format
    if motion_data.get("type") == "motion_start":
        motion_name = motion_data.get("motion")
        facial_expression = motion_data.get("facial_expression", "normal")
        start_time = motion_data.get("start_time", 0)
        
        logger.info(f"🎭 Facial Expression: {facial_expression}")
        logger.info(f"🕺 Motion triggered: {motion_name} at {start_time:.2f}s")
        
        # Trigger motion immediately (it's already timed correctly)
        trigger_motion_now(motion_name)
        trigger_facial_expression(facial_expression)
    else:
        # Handle old format for backward compatibility
        motions = motion_data.get("motions", [])
        facial_expression = motion_data.get("facial_expression", "normal")

        logger.info(f"😊 Facial Expression: {facial_expression}")
        trigger_facial_expression(facial_expression)

        base_time = time.time()
        for motion in motions:
            name = motion.get("name")
            start_sec = motion.get("start", 0)

            def show_motion(motion_name=name):
                logger.info(f"🕺 Motion now: {motion_name}")

            delay = max(0, start_sec + MOTION_COMPENSATION_OFFSET)
            logger.info(f"📆 Scheduling '{name}' in {delay:.2f}s")
            threading.Timer(delay, show_motion).start()

def trigger_motion_now(motion_name):
    logger.info(f"🕺 Motion executing now: {motion_name}")
    # Add your motion triggering logic here
def trigger_facial_expression(expression: str):
    logger.info(f"🎭 Triggering facial expression: {expression}")
    # 👇 Add your actual logic here (e.g. WebSocket, Unity event, OSC)

def motion_listener():
    consumer = KafkaConsumer(
        KAFKA_TOPIC_MOTION,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    logger.info("🚀 Motion listener started.")

    for message in consumer:
        motion_data = message.value
        logger.info(f"📥 Received motion data: {motion_data}")
        trigger_motions(motion_data)

if __name__ == "__main__":
    motion_listener()
