import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskScoliosisResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskScoliosisResNet, self).__init__()

        # استخدام ResNet-34 كعمود فقري بصري قوي
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # طبقة ميزات مشتركة مع BatchNorm و Dropout لحماية النموذج من Overfitting
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # 1. رأس الانحدار لزوايا كوب المنمطة (UT, MT, Lumbar, Max)
        self.cobb_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        # 2. رأس تصنيف الشدة (Healthy, Mild, Moderate, Severe)
        self.intensity_head = nn.Linear(512, 4)

        # 3. رأس تصنيف الدرجة (0-derece, 1-derece, 2-derece, 3-derece)
        self.stat_head = nn.Linear(512, 4)

        # 4. رأس تصنيف الشكل الهندسي (Normal, C, S)
        self.shape_head = nn.Linear(512, 3)

        # 5. رأس تصنيف الموقع (None, Upper Thoracic, Thoracic, Lumbar)
        self.location_head = nn.Linear(512, 4)

    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.shared_fc(features)

        return {
            "cobb_angles": self.cobb_head(shared_features),
            "intensity": self.intensity_head(shared_features),
            "stat": self.stat_head(shared_features),
            "shape": self.shape_head(shared_features),
            "location": self.location_head(shared_features)
        }

if __name__ == "__main__":
    print("جاري اختبار معمارية النموذج المحدثة (Multi-Task ResNet-34)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] الجهاز المستخدم للاختبار: {device}")

    model = MultiTaskScoliosisResNet(pretrained=True).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    outputs = model(dummy_input)

    print("\n تم تجربة واختبار معمارية النموذج بنجاح!")
    print(f" أبعاد مخرج زوايا كوب (Cobb Angles Output): {outputs['cobb_angles'].shape}")
    print(f" أبعاد مخرج الشدة (Intensity Output): {outputs['intensity'].shape}")
    print(f" أبعاد مخرج الدرجة (Stat Output): {outputs['stat'].shape}")
    print(f" أبعاد مخرج الشكل (Shape Output): {outputs['shape'].shape}")
    print(f" أبعاد مخرج الموقع (Location Output): {outputs['location'].shape}")