import os
# Disable warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint

import torch
from torch.utils import data
from torchmetrics import classification, MetricCollection
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import v2 

import pathlib
import time
from datasets import load_dataset

# Logging
import logging
import colorlog
# Plotting
import matplotlib
import matplotlib.pyplot as plt
# QAT

from torch.ao.quantization.pt2e.export_utils import (
    _move_exported_model_to_eval as move_exported_model_to_eval,
    _move_exported_model_to_train as move_exported_model_to_train,
)

from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

import copy
import sys

# QAT flow: 
# 1. Train the model first, save to ckpt.  Usually training a quantized model FROM scratch does not produce good accuracy
# 2. Load model from trained model. Specify config and PREPARE the model for QAT.
# 3. Train the model on GPUs --> QAT in quantized floating point
# 4. Put the model back onto the CPU, now CONVERT into quantized INTEGER model
 
# Logging
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)

# Create a handler
handler = colorlog.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a colored formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

# Add formatter to handler
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

class_names = [
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
]
labels = []
label2id, id2label = dict(), dict()
ckpt_path = "my_checkpoints/best_model_pre_QAT.ckpt"
QAT_ckpt_path = "output_QAT/QAT_Model.ckpt"

def measure_inference_latency(model,
                              device,
                              QAT,
                              input_size=(1, 3, 320, 320),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    if(QAT):
        move_exported_model_to_eval(model)
    else:
        model.eval()
    
    x = torch.rand(size=input_size).to(device)
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave        
def get_model_memory_size(model):
    total_params = 0
    total_bytes = 0
    for param in model.parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size() # numel() is total elements, element_size() is bytes per element
    for buffer in model.buffers():
        total_params += buffer.numel() # Buffers also contribute to total elements
        total_bytes += buffer.numel() * buffer.element_size()
    # Convert bytes to MB for readability
    total_mb = total_bytes / (1024 * 1024)
    return total_mb, total_params

class MobileNetV3(L.LightningModule):
    def __init__(self, num_classes, QAT_trained=False, batch_size=2, lr=0.001):
        super().__init__()
        model = mobilenet_v3_large(weights='DEFAULT')
        num_in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(
            in_features=num_in_features,
            out_features=num_classes
        )
        self.model = model
        self.QAT = QAT_trained
        self.num_classes = num_classes        
        self.batch_size = batch_size
        self.lr = lr
        self.save_hyperparameters()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.valid_metrics = MetricCollection(
            {
                "accuracy": classification.Accuracy(task="multiclass", num_classes=num_classes, average='micro'),
            },
           prefix="val_"
        )
        self.train_metrics = self.valid_metrics.clone(prefix="train_")
        # Max number of points within a validation plot is num_valid_plots / check_val_every_n_epoch
        self.num_valid_plots = 25
        self.valid_hist = []
    def classify_image(self, img, label_true, predicted_label, score, batch_idx):
        # Plot the picture per validation, with the image label
        max_images = 50
        output_image = img.to(torch.device('cpu'))
        if batch_idx <= max_images:
            # val_imgs/QAT
            if(self.QAT):
                dir_name = f"QAT/val_epoch_{self.current_epoch}"
            # val_imgs/non_QAT
            else:
                dir_name = f"non_QAT/val_epoch_{self.current_epoch}"
            dir_path = f"./val_imgs/{dir_name}"
            dir_path = pathlib.Path(dir_path)
            try:
                dir_path.mkdir(parents=True)
            except FileExistsError:
                pass      
        
            plt.figure(figsize=(12, 12))
            plt.imshow(output_image.permute(1, 2, 0))
            plt.text(50, 50, id2label[str(label_true.item())], color='blue', fontsize=14, ha='center', va='center')
            plt.text(50, 100, id2label[str(predicted_label.item())], color='red', fontsize=14, ha='center', va='center')
            plt.text(50, 150, str(score[0].item()) , color='orange', fontsize=14, ha='center', va='center')
            plt.savefig(f"{dir_path}/pred_batch_{batch_idx}")
            plt.close()
        return
    def forward(self, images):
        return self.model(images) 
    def training_step(self, batch, batch_idx):
        img = batch['pixel_values']
        labels = batch['labels']
        outputs = self.model(img)
        _, predicted = torch.max(outputs, 1)
        loss = self.loss_fn(outputs, labels)
        batch_value = self.train_metrics(predicted, labels)
        self.log("train_accuracy", batch_value["train_accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def on_train_epoch_start(self):
        if(self.QAT):
            if self.current_epoch > 5:
                # Freeze quantizer parameters
                self.model.apply(torch.quantization.disable_observer)
            if self.current_epoch > 4:
                for n in self.model.graph.nodes:
                    # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
                    # We set the `training` flag to False here to freeze BN stats
                    if n.target in [
                        torch.ops.aten._native_batch_norm_legit.default,
                        torch.ops.aten.cudnn_batch_norm.default,
                    ]:
                        new_args = list(n.args)
                        new_args[5] = False
                        n.args = new_args
                self.model.recompile()
        return
    def on_train_epoch_end(self):
        self.train_metrics.reset()
    def on_train_end(self):
        if(self.QAT):
            self.model = self.model.to(device=torch.device('cpu'))
            example_inputs = (torch.rand(1, 3, 320, 320, device=torch.device('cpu')),)
            prepared_model_copy = copy.deepcopy(self.model)
            quantized_model = convert_pt2e(prepared_model_copy)
            quantized_model = move_exported_model_to_eval(quantized_model)
            et_program = to_edge_transform_and_lower(
                torch.export.export(quantized_model, example_inputs),
                partitioner=[XnnpackPartitioner()],
            ).to_executorch()
            with open("QAT_Model_Actual.pte", "wb") as f:
                f.write(et_program.buffer)
        return
    def on_validation_model_train(self):
        if(self.QAT):
            self.model = move_exported_model_to_train(self.model)
        else:
            self.model.train()
    def on_validation_model_eval(self):
        if(self.QAT):
            self.model = move_exported_model_to_eval(self.model)
        else:
            self.model.eval()
    def validation_step(self, batch, batch_idx):
        img = batch['pixel_values']
        labels = batch['labels']
        outputs = self.model(img) 
        prediction = torch.softmax(outputs, 1)
        scores, predicted_labels = torch.max(prediction, 1)
        self.valid_metrics.update(predicted_labels, labels)
        loss = self.loss_fn(outputs, labels)
        self.classify_image(img[0], labels[0], predicted_labels, scores, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
    def on_validation_epoch_end(self):
        result = self.valid_metrics.compute()
        self.log("val_accuracy", result["val_accuracy"], prog_bar=True, batch_size=self.batch_size)
        # Log every epoch
        if self.current_epoch % self.num_valid_plots == 0:
            self.valid_hist.clear()
        # History 
        self.valid_hist.append(result)
        # Plot
        fig = plt.figure(figsize=(15, 15), layout="constrained")
        ax1 = plt.subplot(1, 1, 1)
        self.valid_metrics.plot(self.valid_hist, ax=ax1, together=True)
        self.valid_metrics.reset()
        
        if(self.QAT):
            dir_path = f"./validation_plots/QAT"
        else:
            dir_path = f"./validation_plots/non_QAT"
        dir_path = pathlib.Path(dir_path)
        try:
            dir_path.mkdir(parents=True)
        except FileExistsError:
            pass        
        fig.savefig(f"{dir_path}/validation_plot_epoch_{self.current_epoch}.png")
        plt.close(fig)
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0005)    
    
class DogsDataModule(L.LightningDataModule): 
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
    @staticmethod    
    def transforms(examples):
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.RandomPhotometricDistort(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomResizedCrop(320), 
            ]
        )
        examples["pixel_values"] = [image_transform(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples    
    @staticmethod    
    def transforms_valid(examples):
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        examples["pixel_values"] = [image_transform(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples    
    @staticmethod
    def collate_fn(examples):
        images = []
        labels = []
        for example in examples:
            images.append((example["pixel_values"]))
            labels.append(example["label"])
        pixel_values = torch.stack(images)
        labels = torch.tensor(labels)
        return {"pixel_values": pixel_values, "labels": labels}
    @staticmethod
    def worker_init_fn(random_seed):
        """Set the worker ID for reproducibility."""
        import random
        import numpy as np
        import torch

        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        return
    def setup(self, stage=None):
        main_dataset = load_dataset("Donghyun99/Stanford-Dogs", split='train')
        dataset_valid = load_dataset("Donghyun99/Stanford-Dogs", split='test')
        labels = main_dataset.features["label"].names
        index_labels = [labels.index(class_name) for class_name in class_names]
        main_dataset = main_dataset.filter(lambda example: example["label"] in index_labels)
        dataset_valid = dataset_valid.filter(lambda example: example["label"] in index_labels)
        old_to_new_label_map = {old_label: new_label for new_label, old_label in enumerate(index_labels)}
        def remap_labels(example):
            example["label"] = old_to_new_label_map[example["label"]]
            return example
            
        # Apply the remapping
        main_dataset = main_dataset.map(remap_labels)
        dataset_valid = dataset_valid.map(remap_labels)
        
        # split into train and test 
        main_dataset = main_dataset.shuffle(seed=5)
        main_dataset = main_dataset.with_transform(self.transforms)
        dataset_valid = dataset_valid.with_transform(self.transforms_valid)
        self.train_set, self.valid_set = main_dataset, dataset_valid
        
        for i, label in enumerate(class_names):
            label2id[label] = str(i)
            id2label[str(i)] = label
    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True, shuffle=True, persistent_workers=True,
                               num_workers=self.num_workers, collate_fn=self.collate_fn, worker_init_fn=self.worker_init_fn)
    def val_dataloader(self):
        return data.DataLoader(self.valid_set, batch_size=1, pin_memory=True, shuffle=False, persistent_workers=True,
                               num_workers=self.num_workers, collate_fn=self.collate_fn, worker_init_fn=self.worker_init_fn)
if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    train_QAT = True if sys.argv[1].lower() == "true" else False # Train the float and apply QAT from scratch
    max_epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    accumulate_grad_batches = int(sys.argv[4])
    num_classes = len(class_names)
    lr = 0.0001
    train_float = not train_QAT 
    matplotlib.use('Agg')
    
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the first available CUDA device
        logger.info(f"GPU is available! Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using device: CPU")
#------------------------------------------------------------------Training float model----------------------------------------------------------------------------------------------- 
    dataset_mod = DogsDataModule(batch_size=batch_size)
    # checkpoint save trained model
    early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=5, verbose=True, mode="max")
    early_stop_callback_t = EarlyStopping(monitor="train_loss", patience=10000, check_finite=True, verbose=True, mode="min")

    if train_float:
        # Untrained model
        model = MobileNetV3(num_classes=num_classes, QAT_trained=False, batch_size=batch_size, lr=lr)
        checkpoint_callback = ModelCheckpoint(
            dirpath="my_checkpoints/",          # Directory where checkpoints will be saved
            filename="best_model_pre_QAT",      # Naming convention for files
            monitor="val_accuracy",             # Metric to monitor for saving "best" models
            mode="max",                         
            save_top_k=1,                       # Save the top 1 models with the max val_acc
            save_weights_only=True,             
            enable_version_counter=False,
            verbose=True,                       
        )
        # Trainer
        trainer = L.Trainer(check_val_every_n_epoch=1, accumulate_grad_batches=accumulate_grad_batches, devices=[0],
                        log_every_n_steps=10, min_epochs=0, max_epochs=max_epochs, accelerator="gpu", precision="32-true",
                        callbacks=[early_stop_callback, early_stop_callback_t, checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)])
        # Tuner
        # tuner = Tuner(trainer)
        # lr_finder = tuner.lr_find(model, datamodule=dataset_mod)
        # new_lr = lr_finder.suggestion()
        # model.lr = new_lr
        # tuner.scale_batch_size(model, datamodule=dataset_mod, mode="power")
        trainer.fit(model, dataset_mod)
        checkpoint = torch.load(ckpt_path)
        model = MobileNetV3.load_from_checkpoint(checkpoint_path=ckpt_path) 
        trainer.validate(model, dataset_mod)
        float_mb, float_params = get_model_memory_size(model.model)
        logger.debug(f"Float Model Parameters: {float_params:,}")
        logger.debug(f"Float Model Size: {float_mb:.2f} MB") 
        logger.info(f"Average inference time (not quantized) : {(measure_inference_latency(model.model, torch.device('cpu'), False) * 1000 ):.2f} ms")
#------------------------------------------------------------------Training using QAT----------------------------------------------------------------------------------------------- 
    torch.cuda.empty_cache()
    # Now trained (pre-quantized), load the checkpoint (trained float model)
    model = MobileNetV3.load_from_checkpoint(checkpoint_path=ckpt_path, QAT_trained=True) 
    model.model = model.model.to(device=torch.device('cpu'))
    example_inputs = (torch.rand(1, 3, 320, 320, device=torch.device('cpu')),)
    exported_model = torch.export.export(model.model, example_inputs).module()
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    model.model = prepared_model

    if train_QAT:
        # Checkpoint to save QAT model
        checkpoint_callback = ModelCheckpoint(
            dirpath="output_QAT/",          
            filename="QAT_Model",
            monitor="val_accuracy",                 
            mode="max",                         
            save_top_k=1,                       
            save_weights_only=True,            
            enable_version_counter=False,
            verbose=True,                       
        )
        trainer = L.Trainer(check_val_every_n_epoch=2, accumulate_grad_batches=accumulate_grad_batches, devices=[0],
                    log_every_n_steps=10, min_epochs=0, max_epochs=max_epochs, accelerator="gpu",
                    callbacks=[early_stop_callback, early_stop_callback_t, checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)])
        # train using QAT
        trainer.fit(model, dataset_mod)
        float_mb, float_params = get_model_memory_size(model.model)
        logger.debug(f"Float Model Parameters: {float_params:,}")
        logger.debug(f"Float Model Size: {float_mb:.2f} MB") 
        logger.info(f"Average inference time (post-trained quantized) : {(measure_inference_latency(model.model, torch.device('cpu'), True) * 1000 ):.2f} ms")
#---------------------------------------------------------Convert into Quantized model and lowering------------------------------------------------------------------------------------------- 
    checkpoint = torch.load(QAT_ckpt_path)
    model = MobileNetV3.load_from_checkpoint(checkpoint_path=ckpt_path, QAT_trained=True) 
    model.model = model.model.to(device=torch.device('cpu'))
    exported_model = torch.export.export(model.model, example_inputs).module()
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}    
    prepared_model.load_state_dict(model_weights)
   
    # Now evaluate the model from the saved .ckpt file
    if train_QAT:
        quantized_model = convert_pt2e(prepared_model) 
        quantized_model = move_exported_model_to_eval(quantized_model)
        model.model = quantized_model
        et_program = to_edge_transform_and_lower(
            torch.export.export(quantized_model, example_inputs),
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()
        with open("QAT_Model.pte", "wb") as f:
            f.write(et_program.buffer)

        trainer = L.Trainer(min_epochs=0, max_epochs=max_epochs, accelerator="gpu", enable_progress_bar=True) 
        trainer.validate(model, dataset_mod)
        
        quant_mb, quant_params = get_model_memory_size(quantized_model)
        logger.debug(f"Quantized Model Parameters: {quant_params:,}")
        logger.debug(f"Quantized Model Size: {quant_mb:.2f} MB")
        logger.info(f"Average inference time (quantized) : {(measure_inference_latency(quantized_model, torch.device('cpu'), True) * 1000 ):.2f} ms")
