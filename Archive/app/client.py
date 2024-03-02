import grpc
import protos.asr_pb2 as asr_pb2
import protos.asr_pb2_grpc as asr_pb2_grpc


def run():
    # Open a gRPC channel
    channel = grpc.insecure_channel('localhost:5051')

    # Create a stub (client)
    stub = asr_pb2_grpc.ASRStub(channel)

    # Create a new ASR request iterator
    def request_iterator(audio_file):
        with open(audio_file, 'rb') as f:
            while True:
                chunk = f.read(113644)
                if len(chunk) == 0:
                    break
                yield asr_pb2.ASRRequest(audio_chunk=chunk)

    responses = stub.StreamASR(request_iterator('./audio/output_8k.wav'))

    for response in responses:
        print("Received message: ", response.transcript)

if __name__ == '__main__':
    run()

