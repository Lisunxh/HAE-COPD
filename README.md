# HAE

HAE model demo, pretraining, fine-tuning and evaluation.

Tested with Python 3.10.20 and PyTorch 2.8.0 (CUDA 12.8), on CPU and GPU.

`run.py`  Run a forward pass with one random input of shape `(1, 2, 4096)`.

`runtime.py`  Configure cache paths and PyTorch settings; restrict Python file writes to this folder. cuDNN TF32 is enabled; matrix-multiplication TF32 is disabled.

`requirements.txt`  Python dependencies. 

`dataloader/dataset.py`  Dataset and batch preparation. 

`dataloader/augmentation.py`  Signal augmentation. 

`models/hae.py`  HAE encoder, decoder and masking modules. 

`models/classifier.py`  Classification model and prediction head. 

`models/embeddings.py`  Positional and physiological phase embeddings. 

`models/reconstruction.py`  Reconstruction metrics. 

`dataloader/__init__.py`, `models/__init__.py`  Python package markers. 

`train.py`  Five-fold response-task pretraining and classification fine-tuning.

`test.py`  Internal held-out and external evaluation using saved thresholds.

`training_utils.py`  Focal loss, EMA validation, early stopping and classification metrics.


```bash
pip install -r requirements.txt
python -B run.py
```

`.runtime/` is a disposable cache folder recreated when running the demo. The demo checks model execution.

Training requires trusted, already-preprocessed NPY dictionaries, not raw waveform exports. Required fields: `data_array` (N,2,4096), `phase_array` (N,4096), `demographic_normalized_array` (N,5), `impedance_array` (N,8), `label_array` (N,K; binary label in column 0), `result_array` (N,R), `idx_array` (N), and `stratify` (one value per recording). Record indices must be consecutive from zero.

```bash
python -B train.py --internal /path/to/internal.npy --output outputs/train
python -B test.py --checkpoints outputs/train --internal /path/to/internal.npy --external /path/to/external.npy --output outputs/test
```
