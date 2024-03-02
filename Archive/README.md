# Steps to build the docker (1-GPU Required) (No envs required)

1. Clone the repo
2. Build the docker 

```shell
docker build --tag ai-asr-online --file Dockerfile .
```

3. Run the docker

```shell
docker run -dit --gpus 1 --env-file .env --name ai-asr-online ai-asr-online
```

### Convert proto files

```shell
python3 -m grpc_tools.protoc --proto_path=./protos ./protos/asr.proto --python_out=./protos/ --grpc_python_out=./protos/
```
