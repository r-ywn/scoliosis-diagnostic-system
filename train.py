import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_dataloaders
from model import MultiTaskScoliosisResNet

# 1. إعدادات التدريب العامة (Hyperparameters)
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_scoliosis_model.pth"


def train_one_epoch(model, dataloader, optimizer, criterion_mse, criterion_ce):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(DEVICE)
        cobb_gt = batch["cobb_angles"].to(DEVICE)
        intensity_gt = batch["intensity"].to(DEVICE)
        stat_gt = batch["stat"].to(DEVICE)
        shape_gt = batch["shape"].to(DEVICE)
        location_gt = batch["location"].to(DEVICE)

        optimizer.zero_grad()

        # التنبؤ المباشر
        outputs = model(images)

        # حساب خسائر المهام الخمسة
        loss_cobb = criterion_mse(outputs["cobb_angles"], cobb_gt)
        loss_intensity = criterion_ce(outputs["intensity"], intensity_gt)
        loss_stat = criterion_ce(outputs["stat"], stat_gt)
        loss_shape = criterion_ce(outputs["shape"], shape_gt)
        loss_location = criterion_ce(outputs["location"], location_gt)

        # دمج الخسائر في دالة هدف واحدة
        combined_loss = loss_cobb + loss_intensity + loss_stat + loss_shape + loss_location

        combined_loss.backward()
        optimizer.step()

        total_loss += combined_loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(f"   الدفعة [{i + 1}/{len(dataloader)}] - الخسارة الحالية: {combined_loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion_mse, criterion_ce):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            cobb_gt = batch["cobb_angles"].to(DEVICE)
            intensity_gt = batch["intensity"].to(DEVICE)
            stat_gt = batch["stat"].to(DEVICE)
            shape_gt = batch["shape"].to(DEVICE)
            location_gt = batch["location"].to(DEVICE)

            outputs = model(images)

            loss_cobb = criterion_mse(outputs["cobb_angles"], cobb_gt)
            loss_intensity = criterion_ce(outputs["intensity"], intensity_gt)
            loss_stat = criterion_ce(outputs["stat"], stat_gt)
            loss_shape = criterion_ce(outputs["shape"], shape_gt)
            loss_location = criterion_ce(outputs["location"], location_gt)

            combined_loss = loss_cobb + loss_intensity + loss_stat + loss_shape + loss_location
            total_loss += combined_loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    print(f" بدء عملية التدريب على الجهاز: {DEVICE}")

    # جلب داتا التدريب والاختبار
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=2)

    # تهيئة النموذج ونقله لكارت الشاشة
    model = MultiTaskScoliosisResNet(pretrained=True).to(DEVICE)

    # دوال الخسارة
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    # المحسّن
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\n==================== Epoch [{epoch + 1}/{EPOCHS}] ====================")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_mse, criterion_ce)
        val_loss = validate(model, test_loader, criterion_mse, criterion_ce)

        print(f" نتيجة Epoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        # حفظ النموذج عند تحقيق أقل نسبة خسارة على بيانات الاختبار
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" تم حفظ النموذج الأفضل في المسار: {SAVE_PATH}")

    print("\n اكتملت عملية التدريب بنجاح!")