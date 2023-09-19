# Set the name of the Docker image and container
DOCKER_IMAGE_NAME = tensorflow-gpu-masa

# Set the host folder to mount to the Docker container
HOST_FOLDER = $(CURDIR)

.PHONY: all build run

all: build run

build-tensorflow:
	docker build -t $(DOCKER_IMAGE_NAME) -f Docker/MASADockerfile .;

run-tensorflow:
	@docker run --rm -it --gpus all -v $(HOST_FOLDER):/app -p 8888:8888 $(DOCKER_IMAGE_NAME) jupyter lab  --ip='*' --port=8888 --allow-root

build-torch:
	docker build -t torch-gpu-masa -f Docker/MASADockerfileTorch .;

run-torch:
	@docker run --rm -it --gpus all -v $(HOST_FOLDER):/app --shm-size 22G -p 8887:8887 torch-gpu-masa jupyter lab  --ip='*' --port=8887 --allow-root 