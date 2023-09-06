# OCR Task

This repo contains the code for Optical Character Recognition (OCR) for dates written in Arabic numbers. The code is written using Tensorflow.

**Note:** The `ArabicDatesVocabulary` class creates a hash table for mapping arabic characters to labels. For some reason my Windows device did not encode arabic characters despite working fine on Google Colab. So, I guess this is a device specific problem.

## Setup

1. Create virtual environment
```sh
$ python -m venv .venv
# Linux
$ source .venv/bin/activate
# Windows
$ .venv\Scripts\activate
```


2. Install dependencies
```sh
$ pip install -r requirements.txt
```

## Training the Model

A training script is provided for training on your own data. Additionally, you can provide your own checkpoints.

```sh
$ python ocrnet/train.py --model=./models/model --train_dataset=./dataset/ --output_path=./ocr_model/
```

## Inferencing

There is a script provided for inference on your own data. Additionally, you can provide your own checkpoints.

```sh
$ python ocrnet/inference.py --model=./models/model --test_dataset=./dataset/
```

## Model Conversion

You can convert the model to ONNX using the command below.

```sh
$ python -m tf2onnx.convert --saved-model ./models/model --output ./models/model.onnx
```

To convert to TensorRT, install `tensorrt` then, run the command below.
```sh
$ python ocrnet/converter.py --input_name=./models/model --ouptut=./models/trt_model
```