# object-detection

Object detection app, uses facebook/detr-resnet-50 to detect objects and outputs an the sample image with the bounding boxes and labels on top of the original image.

The docker compose setup will use a local directory mount to store the model to persist across container runs.
# Run
```bash
docker-compose build
docker-compose up
```

# DEV
Enable the dev "comand" for the object-detection service in the docker-compose.yaml.
From here, you can use dev-containers in order to connect to the running container and make any changes from there.
From a new terminal you can run
```bash
python3 src/app.py
```