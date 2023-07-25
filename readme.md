# Image Filter FastAPI AI Application

![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-orange)

This is a FastAPI application that utilizes a deep learning model to classify images for adult content. The application uses TensorFlow and the InceptionResNetV2 model fine-tuned for NSFW classification.

## Features

- Classify images for adult content using TensorFlow and InceptionResNetV2 model.
- Utilizes Docker for easy deployment and isolation.
- FastAPI web framework for handling image classification requests.

## Requirements

- Docker (for running the application in a container)
- Python 3.6 or higher (for development)

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your_username/image-filter-app.git .
``` 
2. Install the required packages:

```bash
pip install -r requirements.txt
``` 
3. Run the application using Docker:

```bash
docker build -t image_filter_app .
```
```bash
docker run -d -p 80:80 --name image_filter_container image_filter_app
``` 
### The application will be accessible at http://localhost/classify/ as POST for image classification.

Request Body:

```bash
{
  "image_url": "https://example.com/path/to/image.jpg"
}

```
Response:

```bash
{
  "is_adult_content": true
}
