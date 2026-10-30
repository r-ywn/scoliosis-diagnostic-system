import os
import matplotlib.pyplot as plt

BASE_DIR = r"D:\Scoliosis_Project"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# البيانات الحقيقية الفعلية المستخرجة من سجل تدريب الـ 30 Epochs
epochs = list(range(1, 31))

train_loss = [
    1.0482, 0.8277, 0.7572, 0.7343, 0.6899, 0.6559, 0.6390, 0.6307, 0.6018, 0.5872,
    0.5555, 0.5564, 0.5209, 0.5087, 0.4781, 0.4560, 0.4213, 0.4097, 0.3773, 0.3630,
    0.3364, 0.3185, 0.2934, 0.2784, 0.2639, 0.2497, 0.2388, 0.2224, 0.2260, 0.2246
]

train_acc = [
    65.97, 73.44, 75.86, 76.32, 77.92, 79.15, 79.97, 80.15, 81.17, 81.41,
    82.71, 82.65, 84.19, 84.26, 85.46, 86.44, 87.85, 88.44, 89.52, 90.37,
    91.36, 91.79, 93.06, 93.46, 93.68, 94.52, 94.91, 95.47, 95.17, 95.45
]

val_acc = [
    71.55, 73.08, 74.59, 75.41, 76.58, 79.62, 78.88, 81.50, 78.06, 79.68,
    76.21, 78.68, 81.04, 84.37, 79.56, 83.37, 85.53, 83.23, 84.25, 86.44,
    86.10, 85.53, 85.42, 86.53, 86.61, 87.55, 87.35, 87.35, 87.89, 87.27
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', lw=2, ms=4)
ax1.set_title("Real Training Loss Progression (30 Epochs)")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.plot(epochs, train_acc, 'b-o', label='Train Acc (%)', lw=2, ms=4)
ax2.plot(epochs, val_acc, 'g--^', label='Val Acc (%)', lw=2, ms=4)
ax2.set_title("Real Training & Validation Accuracy (%)")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, "loss_and_accuracy_curves.jpg")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"[+] تم توليد وحفظ منحنيات التدريب والتحقق الحقيقية بنجاح في: {save_path}")