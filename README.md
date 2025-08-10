# Quantization-Aware Training for Mobile Deployment: Deploying MobileNetV3 on Android

Using the PyTorch Lightning, I applied [PyTorch 2 Export Quantization Aware Training](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_qat.html#) on the MobileNetV3 and achieved a 4x reduction of memory footprint from 16.18 MB of the float model to 4.2 MB quantized. Quantized model is then exported via [ExecuTorch](https://docs.pytorch.org/executorch/stable/getting-started.html) via the [XNNPACK](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html#target-requirements), ideal for Android and consequently mobile ARM processors.

This is trained using HuggingFace datasets on [Donghyun99/Stanford-Dogs](https://huggingface.co/datasets/Donghyun99/Stanford-Dogs) on the following subset of classes:
```
"komondor",
"German_shepherd",
"toy_poodle",
"pug",
"Yorkshire_terrier",
"Doberman",
"Bernese_mountain_dog",
"French_bulldog",
"chow",
"Chihuahua",
"Eskimo_dog",
```

The hardware used to train the model is on the NVIDIA GeForce RTX 3050 Ti Laptop GPU, and the phone I use to deploy my model is the Galaxy Note 2. The quantized model is located inside the **assets** folder of ImageClassificationQAT. 

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation and Running](#installation-and-running)
- [Code Layout](#code-layout)
- [Journey](#journey)
- [Resources](#resources)

## Prerequisites
Due to ExecuTorch, running Python script to train your model requires the following requirements,

- Python 3.10+ 
- Linux or MacOS (x86 or ARM) or WSL (Windows)

To deploy your model onto Android, download the following:
- [Android Studio Meerkat (2024.3.2 Patch 1)](https://developer.android.com/studio/archive)


## Installation and Running
Only the Python script needs to run inside a Linux environment. I copied the "Python" folder into WSL.

1.  Clone the repository:
    `git clone https://github.com/vinle0/MobileNetV3-QAT`
2.  Navigate to the project directory:
    `cd MobileNetV3-QAT`

### Quantization

1. Create a virtual environment
    ```bash
    > cd Python
    > python3 -m venv env
    > source env/bin/activate
    > pip install -r requirements.txt
    ```
2. Run the script to train the float model and applies QAT. It is split into two segments of training because the CPU/GPU would have an out-of-memory error if done all at once. Inside the script file, you can change the batch size, accumulate batch size (part of PyTorch Lightning to achieve larger batch sizes without overloading memory), and number of epochs. The script is the training parameters that I used, but feel free to change the parameters (including inside the python file).
    ```bash
    > .\script.sh
    ```
3. This exports QAT_Model_Actual.pte and QAT_Model.pte. Functionally, there is no difference, as the QAT model is converted into a quantized model to be used in Android. After validating QAT_Model_Actual.pte, it achieved a validation accuracy of 82% and is currently inside the "ImageClassificationQAT" folder. The QAT model is copied into the "assets" folder like so
    ```bash
    > cp QAT_Model_Actual.pte ../ImageClassificationQAT/app/src/main/assets
    ```
### Deployment on Android
Deploying on Android requires hardware or simulator to have at least the following requirements for [XNNPACK](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html#target-requirements):

- ARM64 on Android, iOS, macOS, Linux, and Windows.
- x86 and x86-64 (up to AVX512) on Windows, Linux, macOS, Android, and iOS simulator.
- Further, the [Android Library (AAR)](https://docs.pytorch.org/executorch/stable/using-executorch-android.html#android-backends) requires arm64-v8a and x86_64 architecture.

I personally used the Pixel 4 as a simulator when debugging my application, but any Android phone that fits the above requirements is fine.

1. Open the project "ImageClassificationQAT" from Android Studio 
2. Open up a terminal, and then type 
`adb push app/src/main/assets/QAT_Model_Actual.pte /data/local/tmp/`
to push the model onto the phone
3. Run the Android Studio code to download the app onto the phone
4. Open up the app ImageClassificationQAT, accept the permissions to use the front camera, and now it is ready for use! 

## Code Layout
The code is split into the training of the model (Python folder) and the deployment on Android (ImageClassificationQAT)

### Python QAT
Inside the Python folder contains the script to train the model using QAT. It will output two folders: the validation plots (containing the validation accuracy of the model, using matplotlib) and the validation images (containing true label, predicted label, and confidence score) to verify if the model is outputting correctly. PyTorch Lightning simplifies the actions on the model by modularly making functions to train and validate. Within PyTorch Lightning, it is easy to create your dataset and model and then simply call `.train()` or `.validate()` on the model. It contains a customized logger to debug any tensors or values inside the application.

The HuggingFace dataset is the Standford Dogs dataset, where inside the setup for the DogsDataModule, I filter out the labels and their indexes to the classes that I want. 

It first trains MobileNetV3 from scratch to achieve a model that is trained on the dataset, and it saves the most accurate model (based on validation accuracy) to `my_checkpoints` folder. Then the I load in the checkpoint and then perform QAT with PT2E, also exporting the most accurate mdoel to the `output_QAT` folder as a prepared model. Finally, the QAT model is then quantized and exported using XNNPACK for ARM CPUs.

### Android
This requires the AAR library for Android, where it is included in the build.gratle.kts using Maven. It utilizes CameraX of the front camera and requests permissions to use it. The model is used through the image analysis mode, outputting the result through text.

There is a TensorImageUtils.java from the [ExecuTorch on Android example](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo) to convert our bitmap into tensors for our model.

## Journey 
Essentially, I was first exploring how to interface PyTorch Lightning with training a model, as I was unfamiliar with the technology. The main roadblock for me was testing which of the TorchVision models are able to not only be quantized but also be exported to Android. Although already optimized for edge devices, I chose MobileNetV3 to see if it could be quantized further and still maintain good accuracy. Also, its lower amount of parameters leads to less time for training.

### Challenge: Object Detection CV Model on ASL Dataset
Initially, I was exploring doing object detection (with bounding boxes) with the FasterRCNN model. Training on the ASL dataset (downloaded [locally](https://public.roboflow.com/object-detection/american-sign-language-letters)) to recognize sign language, I plot out the confusion matrix and mAP for it. Since the [model](https://github.com/pytorch/pytorch/issues/146152) is not export-compliant with `torch.export()`, I could neither use PT2E nor TensorFlow Lite (.tflite under Google AI Edge). I attempted previous quantizations like Eager Mode. This was done through the [Intel Neural Compressor QAT](https://intel.github.io/neural-compressor/latest/docs/source/quantization.html#accuracy-aware-tuning) 2.0 API, which appears to succesfully quantize the model:

```python
# Here is a snippet of the code used to quantize my LightningModule
model = LightningFasterRCNN_MobileNetv3(num_classes=num_classes, QAT_trained=False, batch_size=batch_size, lr=lr)
dataset_mod = COCO2017DataModule(batch_size=batch_size)

...

from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.training import prepare_compression
model.model.to(torch.device("cpu"))
model.model = copy.deepcopy(model.model)
conf = QuantizationAwareTrainingConfig()
compression_manager = prepare_compression(model.model, conf)
compression_manager.callbacks.on_train_begin()
model.model = compression_manager.model
# train using QAT
trainer.fit(model, dataset_mod)
trainer.validate(model, dataset_mod)
compression_manager.callbacks.on_train_end()
# saves a .pt file
compression_manager.save("./output_QAT")
```
While it achieved an acceptable accuracy (~70%), the problem still arises with exporting the model, as the Intel API doesn't provide a function to convert the model optimized for mobile. I tried to use the previous version (replaced by ExecuTorch) of PyTorch Mobile with the export of `torch.jit.script(model)` according to this [link](https://medium.com/@adrian.errea.lopez/from-pytorch-model-to-mobile-application-50bc5729ed83). Since the FasterRCNN model was more recent than PyTorch Mobile, I suspect the model was not supported and thus resulted in a faulty .ptl file for mobile. Indeed, the subsequent Android app couldn't read the model, resulting in no outputs of the model. As such, I decied to move onto the simpler CV application of image classification.

### Challenge: Working with HuggingFace Datasets
I had previous experience with locally installed datasets, and it is my first time working with HuggingFace. Being open-source, there are a few datasets that fit with my model regarding image size and amount of data in the datasets. I first tried doing the Food101 dataset but that had the some faulty labels that didn't allow me to filter classes that I want. Following the various [tutorials](https://huggingface.co/docs/datasets/en/image_classification) to interface with HuggingFace, I was able to successfully integrate it with PyTorch Lightning.  

### Takeaways
Overall, this was a lesson in learning new technologies and being to learn from my failures. Even if object detection didn't work out, I was still able to transfer my knowledge over to image classification, showing just how adaptable PyTorch Lightning is. Additionally, I explored  different technologies like the Intel Neural Compressor, Google AI Edge, and the ONNX format with some success on FasterRCNN. I learn how to debug my models, where I ensure that my image data is succesfully converted into the proper tensor format (done through PyTorch transforms). This is especially apparent within the Android application, where I had to similarily convert ImageProxy images to bitmaps to tensors! 

Performing QAT requires more epochs than training on a float model, namely because it requires observers to learn the quantization parameters over a longer period of time. The biggest lesson, then, is learning how PyTorch Lightning and PyTorch Quantization interacts with each other, allowing a seamless integration to quantize the model.


## Resources
These are list of resources that I used to help me with the project. They explore different solutions to the QAT problem, and not all of them are used for image classification. 

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) and [TorchMetrics with Lightning](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
- [PyTorch 2 Export Quantization Aware Training](https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_qat.html#) and [PyTorch Quantization](https://docs.pytorch.org/docs/stable/quantization.html)
- [PyTorch QAT (Eager Mode)](https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/)
- [(Outdated) PyTorch Export to Mobile](https://tutorials.pytorch.kr/recipes/mobile_interpreter.html#android)
- [Intel Neural Compressor QAT (Torch FX)](https://intel.github.io/neural-compressor/latest/docs/source/quantization.html#accuracy-aware-tuning) and [Post-trained Quantization from PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/post_training_quantization.html#pytorch-lightning-model)
- [ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html) and [XNNPACK for ARM CPUs](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html#)
- [HuggingFace with PyTorch](https://huggingface.co/docs/datasets/en/use_with_pytorch)
- [Interfacing CameraX in Android](https://developer.android.com/codelabs/camerax-getting-started#5) and [Integrating PyTorch with Android](https://www.youtube.com/watch?v=ghxLlsT7ebo)