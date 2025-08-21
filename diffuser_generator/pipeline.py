import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

# Albumentations 라이브러리 임포트
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mvtec_dataset import MVTecDataset
from anomalib.models import Patchcore


def run_full_pipeline():
    # --- 1. 설정 ---
    CATEGORY = 'capsule'

    # 중요: capsule로 학습시킨 모델의 경로를 확인하고 맞게 수정해주세요.
    PATCHCORE_CHECKPOINT_PATH = f"anomalib_results/Patchcore/MVTecAD/{CATEGORY}/v0/weights/lightning/model.ckpt"

    # 중요: capsule 불량으로 학습시킨 분류 모델 경로를 확인하고 맞게 수정해주세요.
    CLASSIFIER_CHECKPOINT_PATH = "./best_defect_classifier.pth"

    ROOT_DIR_MVTEC = 'C:/Users/AI-00/Desktop/mvtec_anomaly_detection'

    IMAGE_SIZE_PATCHCORE = 256
    IMAGE_SIZE_CLASSIFIER = 224
    BATCH_SIZE = 32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 중요: train_classifier.py 실행 시 나왔던 Class mapping과 정확히 일치시켜야 합니다.
    CLASS_NAMES = ['crack', 'faulty_imprint', 'poke', 'rotated', 'scratch', 'squeeze']

    # --- 2. 두 개의 모델 로드 ---
    print(f"Loading Stage 1 (PatchCore) model from: {PATCHCORE_CHECKPOINT_PATH}")
    patchcore_model = Patchcore.load_from_checkpoint(checkpoint_path=PATCHCORE_CHECKPOINT_PATH)
    patchcore_model.to(device)
    patchcore_model.eval()
    print("PatchCore model loaded.")

    print(f"Loading Stage 2 (Classifier) model from: {CLASSIFIER_CHECKPOINT_PATH}")
    classifier_model = models.resnet18(weights=None)
    num_ftrs = classifier_model.fc.in_features
    classifier_model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    classifier_model.load_state_dict(torch.load(CLASSIFIER_CHECKPOINT_PATH, weights_only=True))
    classifier_model.to(device)
    classifier_model.eval()
    print("Classifier model loaded.")

    # --- 3. 테스트 데이터 준비 ---
    patchcore_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE_PATCHCORE, width=IMAGE_SIZE_PATCHCORE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    classifier_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE_CLASSIFIER, IMAGE_SIZE_CLASSIFIER)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = MVTecDataset(
        root_dir=ROOT_DIR_MVTEC,
        category=CATEGORY,
        phase='test',
        transform=patchcore_transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} test images for '{CATEGORY}'.")

    # --- 4. 2단계 파이프라인 직접 실행 ---
    print("\n--- Running Prediction Pipeline ---")
    results_list = []

    progress_bar = tqdm(test_dataloader, desc="Analyzing results")

    with torch.no_grad():
        for data_batch in progress_bar:
            images, true_labels, image_paths, _ = data_batch['image'], data_batch['label'], data_batch['image_path'], \
            data_batch['mask']

            images = images.to(device)

            # 더 이상 임계값(threshold)을 찾지 않도록.
            patchcore_outputs = patchcore_model(images)
            pred_scores = patchcore_outputs.pred_score

            for i in range(len(images)):
                image_path = Path(image_paths[i])
                true_label_idx = true_labels[i].item()
                anomaly_score = pred_scores[i].item()

                true_label_name = "Normal" if true_label_idx == 0 else "Anomaly"

                # 2단계: 모든 이미지에 대해 불량 유형 분류를 수행
                image_pil = Image.open(image_path).convert("RGB")
                image_tensor_cls = classifier_transform(image_pil).unsqueeze(0).to(device)
                outputs_cls = classifier_model(image_tensor_cls)
                _, pred_idx_cls = torch.max(outputs_cls, 1)
                classified_defect_type = CLASS_NAMES[pred_idx_cls.item()]

                results_list.append({
                    "Filename": image_path.name,
                    "Ground Truth": f"{true_label_name} ({image_path.parent.name})",
                    "Anomaly Score": f"{anomaly_score:.4f}",
                    "Stage 2 (Classification)": classified_defect_type,
                })

    # --- 5. 최종 결과 출력 ---
    print("\n" + "=" * 80)
    print("Final Pipeline Results")
    print("=" * 80)
    results_df = pd.DataFrame(results_list)
    results_df_sorted = results_df.sort_values(by="Anomaly Score", ascending=False)
    print(results_df_sorted.to_string())
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()