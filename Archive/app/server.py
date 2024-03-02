import logging
import os
from concurrent import futures

import grpc
import numpy as np
import protos.asr_pb2 as asr_pb2
import protos.asr_pb2_grpc as asr_pb2_grpc
from helpers import OnlineASRProcessor, load_model
from helpers.tokenizer import create_tokenizer
from helpers.utils import test_service

## --------------------------------------------------------------- ##
## -------------------------- Load ENVS -------------------------- ##
## --------------------------------------------------------------- ##

port = int(os.getenv("PORT", "5051"))
size = os.getenv("MODEL", "large-v2")
language = os.getenv("LANGUAGE", "en")
backend = os.getenv("BACKEND", "faster-whisper")
model_cache_dir = os.getenv("CACHE_DIR", None)
model_dir = os.getenv("MODEL_DIR", None)    

## --------------------------------------------------------------- ##
## ------------------------- Init logger ------------------------- ##
## --------------------------------------------------------------- ##

LOGS_FOLDER = "./logs"
os.makedirs(LOGS_FOLDER, exist_ok=True)
logs_file = f"{LOGS_FOLDER}/server.logs"
print("Creating logging file - ", logs_file)
open(logs_file, "w").close()
logging.basicConfig(filename=logs_file, level=logging.INFO, format="%(asctime)s %(message)s")

asr = load_model(size=size, language=language, model_cache_dir=model_cache_dir, model_dir=model_dir)
test_service(demo_audio_path = "./audio/test.wav", asr=asr)
logging.info("Loading Processor...")
online = OnlineASRProcessor(asr, create_tokenizer(language))
logging.info("Loaded Processor")

class ASRService(asr_pb2_grpc.ASRServicer):
    def __init__(self):
        logging.info(f"Server started on port:> {port}")

    def StreamASR(self, request_iterator, context):
        for request in request_iterator:
            hot_words = request.hot_words
            audio_chunk = request.audio_chunk
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16) / 32768  
            try:
                online.insert_audio_chunk(audio_data)
                if transcript := online.process_iter(initial_prompt=hot_words):
                    yield asr_pb2.ASRResponse(transcript=transcript[2])
            except Exception as e:
                print(e)

        if transcript := online.finish():
            yield asr_pb2.ASRResponse(transcript=transcript[2])
        online.init()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    asr_pb2_grpc.add_ASRServicer_to_server(ASRService(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()