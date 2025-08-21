import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

# from torchvision import transforms
from torchvision.datasets import ImageFolder

from mvtec_dataset import MVTecDataset

# Albumentations 라이브러리를 임포트합니다.
import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import ImageBatch
from torchmetrics.classification import BinaryConfusionMatrix

# 추가 데이터셋 로딩을 위한 래퍼 클래스는 여전히 유효합니다.
class AnomalibDatasetWrapper(Dataset):
    def __init__(self, dataset_to_wrap):
        self.dataset_to_wrap = dataset_to_wrap

    def __len__(self):
        return len(self.dataset_to_wrap)

    def __getitem__(self, idx):
        image, _ = self.dataset_to_wrap[idx]
        # Engine이 fit/test에서 사용할 수 있도록 딕셔너리 형태로 반환합니다.
        # DataLoader가 자동으로 텐서로 묶어줍니다.
        return {"image": image, "label": 0}


# 딕셔너리를 ImageBatch 객체로 변환해주는 '번역기' 함수
def custom_collate_fn(batch):
    """
    DataLoader가 생성한 딕셔너리 리스트를 anomalib의 ImageBatch 객체로 변환합니다.
    """
    # batch는 [{'image': tensor, 'label': 0, 'image_path': '...'}, ...] 형태의 리스트입니다.
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    # 'gt_mask' 키워드를 사용하여 ImageBatch에 마스크 전달
    return ImageBatch(image=images, gt_label=labels, gt_mask=masks, image_path=image_paths)


def plot_confusion_matrix(tn, fp, fn, tp, save_path):
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    print(f"\n📊 Confusion Matrix 그래프가 '{save_path}'에 저장되었습니다.")


def main():
    # --- 1. 설정 ---
    ROOT_DIR_MVTEC = 'C:/Users/AI-00/Desktop/mvtec_anomaly_detection'
    CATEGORY = 'capsule'
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    PROJECT_ROOT = Path("./anomalib_results")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. 데이터 준비 ---
    print("Loading datasets with masks...")
    # torchvision.transforms.Compose 대신 A.Compose를 사용합니다.
    transform_pipeline = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # MVTecDataset은 albumentations 변환기를 전달받음
    train_dataset = MVTecDataset(root_dir=ROOT_DIR_MVTEC, category=CATEGORY, phase='train',
                                 transform=transform_pipeline)
    full_test_dataset = MVTecDataset(root_dir=ROOT_DIR_MVTEC, category=CATEGORY, phase='test',
                                     transform=transform_pipeline)

    val_size = int(0.5 * len(full_test_dataset))
    test_size = len(full_test_dataset) - val_size
    val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                             collate_fn=custom_collate_fn)

    # --- 3. 모델 및 엔진 초기화 ---
    print("\nInitializing PatchCore model and Engine...")
    model = Patchcore()
    engine = Engine(default_root_dir=PROJECT_ROOT, accelerator=device, devices=1)
    print("Model and Engine initialized.")

    # --- 4. 모델 학습 ---
    print(f"\n--- Training PatchCore for '{CATEGORY}' category ---")
    engine.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("--- Training complete ---")

    # --- 5. 성능 평가 ---
    print("\n--- Testing on original MVTec test set ---")
    # datamodule 대신 test_loader를 직접 전달합니다.
    test_results = engine.test(model=model, dataloaders=test_loader)
    print("--- Testing and Visualization complete ---")

    # --- 6. 최종 결과 출력 ---
    results_dict = test_results[0]
    print("\n--- Predicting to get raw labels for Confusion Matrix ---")
    # datamodule 대신 test_loader를 직접 전달합니다.
    predictions = engine.predict(model=model, dataloaders=test_loader)

    all_pred_labels = []
    all_gt_labels = []
    all_gt_masks = []
    for batch in predictions:
        all_pred_labels.append(batch.pred_label)
        all_gt_labels.append(batch.gt_label)
        all_gt_masks.append(batch.gt_mask)  # 마스크 결과 수집
    predicted_labels = torch.cat(all_pred_labels)
    ground_truth_labels = torch.cat(all_gt_labels)

    bcm = BinaryConfusionMatrix().to(device)
    conf_mat_tensor = bcm(predicted_labels.to(device), ground_truth_labels.to(device))
    tn, fp, fn, tp = conf_mat_tensor.flatten().int().tolist()
    print("--- Confusion Matrix calculation complete ---")

    total_normal_test = (ground_truth_labels == 0).sum().item()
    total_abnormal_test = (ground_truth_labels == 1).sum().item()

    print("\n" + "=" * 50)
    print("최종 성능 측정 결과")

    # --- 7. 오탐지/미탐지 이미지 분석 ---
    # test_dataset (Subset 객체)에서 경로를 가져옵니다.
    print("\n" + "=" * 50)
    print("오탐지 및 미탐지 상세 분석")
    print("=" * 50)
    test_image_paths = [test_dataset.dataset.image_paths[i] for i in test_dataset.indices]

    fp_indices = torch.where((ground_truth_labels.cpu() == 0) & (predicted_labels.cpu() == 1))[0]
    fn_indices = torch.where((ground_truth_labels.cpu() == 1) & (predicted_labels.cpu() == 0))[0]
    print(f"\n[오탐지된 정상 이미지 (False Positives) - 총 {len(fp_indices)}개]")
    if len(fp_indices) > 0:
        for idx in fp_indices:
            print(f"  - {Path(test_image_paths[idx]).name}")
    else:
        print("  - 없음")
    print(f"\n[미탐지된 불량 이미지 (False Negatives) - 총 {len(fn_indices)}개]")
    if len(fn_indices) > 0:
        for idx in fn_indices:
            print(f"  - {Path(test_image_paths[idx]).name}")
    else:
        print("  - 없음")

    figure_save_path = Path(engine.trainer.log_dir)
    image_save_path = figure_save_path / "images"
    print("\n[분석 방법]")
    print("위 이미지 이름과 동일한 히트맵을 아래 경로에서 찾아 원인을 분석하세요:")
    print(f"➡️  {image_save_path.resolve()}")
    print("=" * 50)

if __name__ == '__main__':
    main()