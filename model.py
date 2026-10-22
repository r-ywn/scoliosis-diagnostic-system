import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskScoliosisResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskScoliosisResNet, self).__init__()

        # 1. تحميل النموذج المسبق التدريب (ResNet-18)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # أخذ عدد ميزات الطبقة الأخيرة قبل إزالتها
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 2. طبقة فهم ميزات مشتركة (Shared Dense Layer)
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 3. رؤوس المخرجات الخمسة (Output Heads)
        self.cobb_head = nn.Linear(256, 4)  # Regression (Upper, Main, Lumbar, Max)
        self.intensity_head = nn.Linear(256, 4)  # Classification (Healthy, Mild, Moderate, Severe)
        self.stat_head = nn.Linear(256, 4)  # Classification (0, 1, 2, 3 derece)
        self.shape_head = nn.Linear(256, 3)  # Classification (Normal, C, S)
        self.location_head = nn.Linear(256, 4)  # Classification (None, Upper, Thoracic, Lumbar)

    def forward(self, x):
        # استخراج الميزات البصرية من العمود الفقري
        features = self.backbone(x)
        shared_features = self.shared_fc(features)

        # التنبؤ بالمخرجات الخمسة متوازية
        cobb = self.cobb_head(shared_features)
        intensity = self.intensity_head(shared_features)
        stat = self.stat_head(shared_features)
        shape = self.shape_head(shared_features)
        location = self.location_head(shared_features)

        return {
            "cobb_angles": cobb,
            "intensity": intensity,
            "stat": stat,
            "shape": shape,
            "location": location
        }


if __name__ == "__main__":
    print("جاري اختبار هيكل النموذج (Multi-Task ResNet-18)...")

    # اختيار الجهاز الحسابي (GPU كارت الشاشة إن وجد)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" تم توجيه النموذج للعمل على: {device}")

    model = MultiTaskScoliosisResNet(pretrained=True).to(device)

    # تجربة إدخال عينة صور وهمية للتحقق من سلامة الأبعاد
    dummy_input = torch.randn(2, 3, 224, 224).to(device)  # دفعة من صورتين
    outputs = model(dummy_input)

    print("\n تم بناء وتجربة هيكل النموذج بنجاح!")
    print(f" أبعاد مخرج زوايا كوب (Cobb Angles Output): {outputs['cobb_angles'].shape}")
    print(f"️ أبعاد مخرج الشدة (Intensity Output): {outputs['intensity'].shape}")
    print(f"️ أبعاد مخرج الشكل (Shape Output): {outputs['shape'].shape}")
    print(f"️ أبعاد مخرج الموقع (Location Output): {outputs['location'].shape}")