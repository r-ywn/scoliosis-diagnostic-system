import torch
import torchvision
import numpy as np
import sklearn
import matplotlib

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print("GPU test:", y.shape)