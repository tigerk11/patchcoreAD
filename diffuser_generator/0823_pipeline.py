# run_final_pipeline.py (최종 버전)

import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import os
import glob
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

from torch.utils.data import DataLoader, TensorDataset
from anomalib.models import Patchcore
from anomalib.data import ImageBatch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typer.rich_utils import FORCE_TERMINAL

# =================================================================================
# --- ### 사용자 설정 ### ---
# =================================================================================
# 1. 학습된 모델 파일 경로
MODEL_SAVE_PATH = Path("./final_patchcore_model.pt")

# 2. 테스트할 데이터셋('capsule' 폴더) 경로
DATA_DIR = Path("C:/Users/AI-00/Desktop/capsule")

# 3. 최종 결과(이미지, 리포트)를 저장할 폴더 경로
RESULTS_SAVE_DIR = Path("./final_pipeline_results")

# 4. 분석 시 사용할 이상 점수 임계값 (결과를 보고 튜닝)
ANOMALY_THRESHOLD = 0.5

ROI_RECT = (67, 330, 881, 324)
IMAGE_SIZE = 256
BATCH_SIZE = 16

# 재학습 하고싶을때 True, 기존 학습한모델로 돌리고싶을땐 False
FORCE_RETRAIN = True
# =================================================================================

def train_model(device):
    """모델 학습을 위한 함수"""
    print("\n" + "=" * 60)
    print("--- 1. 모델 학습 시작 ---")
    print("=" * 60)

    # transform_roi = A.Compose([
    #     A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    #     A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
    #     A.Blur(blur_limit=3, p=0.5),
    #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2(),
    # ])

    ##
    """
    brightness : 이미지의 전체적인 밝기, 
    contrast : 대비 / 이미지의 대비(어두운 부분과 밝은 부분의 차이)
                값이 낮아지면 이미지가 전반적으로 회색처럼 평평해지고, 높아지면 명암이 더욱 뚜렷해집니다. 
                이를 통해 희미한 조명이나 강한 조명 환경에 대한 내성을 기를 수 있습니다.
    saturation : 채도 
                값이 0에 가까워질수록 흑백 사진처럼 되고, 높아질수록 색상이 매우 진하고 선명해집니다. 
                제품 원료의 미세한 색상 차이나 카메라 센서의 색 표현 차이를 극복하는 데 도움을 줍니다.
    hue : 색조 
    p : 적용확률 
        0.5는 학습에 들어가는 이미지 100장 중 50장 정도는 색상 변경이 적용되고, 
        나머지 50장은 원본 그대로 학습에 사용됩니다.
    """
    transform_roi = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),

        A.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.3, hue=0.1, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- [핵심 수정] 데이터 사전 로딩 (Pre-loading) ---
    print("학습 이미지를 메모리로 사전 로딩 중...")
    train_image_paths = sorted(glob.glob(str(DATA_DIR / "train" / "good" / "*")))
    train_roi_tensors = []
    for img_path in tqdm(train_image_paths, desc="Pre-loading train images"):
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        x, y, w, h = ROI_RECT
        roi_image_np = image_rgb[y:y + h, x:x + w]

        transformed = transform_roi(image=roi_image_np)
        train_roi_tensors.append(transformed["image"])

    # 사전 로딩된 텐서들로 TensorDataset 생성
    train_dataset = TensorDataset(torch.stack(train_roi_tensors))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"사전 로딩 완료. {len(train_dataset)}개의 ROI로 학습을 시작합니다.")

    model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"])
    model.to(device)

    model.on_train_epoch_start()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="특징 추출 중"):
            # TensorDataset은 텐서를 튜플로 반환하므로, 첫 번째 요소를 사용
            roi_images_tensor = batch[0].to(device)
            training_batch = ImageBatch(image=roi_images_tensor)
            model.training_step(training_batch, 0)

    print("메모리 뱅크 학습 중...")
    model.on_train_epoch_end()

    torch.save({
        'model_state_dict': model.state_dict(),
        'roi_rect': ROI_RECT
    }, MODEL_SAVE_PATH)
    print(f"모델 학습 완료. '{MODEL_SAVE_PATH}'에 저장되었습니다.")

def analyze_results(device):
    """학습된 모델로 분석 및 리포트 생성을 위한 함수"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("--- 분석 및 시각화 시작 ---")
    print(f"사용 디바이스: {device}")
    print("=" * 60)

    if not MODEL_SAVE_PATH.exists():
        print(f"[!!!] 모델 파일을 찾을 수 없습니다: {MODEL_SAVE_PATH}")
        return

    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=True)
    model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    ROI_RECT = checkpoint['roi_rect']
    print(f"모델 로딩 완료. 학습된 ROI: {ROI_RECT}")

    transform_roi = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- 테스트 데이터 사전 로딩 ---
    print("테스트 이미지를 메모리로 사전 로딩 중...")
    test_image_paths = sorted(glob.glob(str(DATA_DIR / "test" / "**" / "*"), recursive=True))
    test_data = []
    for img_path_str in tqdm(test_image_paths, desc="Pre-loading test images"):
        img_path = Path(img_path_str)
        if not (img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']): continue

        image_bgr = cv2.imread(img_path_str)
        if image_bgr is None:
            print(f"경고: {img_path.name} 파일을 읽을 수 없습니다.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        gt_defect_type = img_path.parent.name
        gt_label = 0 if gt_defect_type.lower() == 'good' else 1

        test_data.append({
            "path": img_path, "full_image": image_rgb,
            "gt_label": gt_label, "gt_defect": gt_defect_type,
        })
    print(f"사전 로딩 완료. {len(test_data)}개의 테스트 이미지를 분석합니다.")

    results = []
    with torch.no_grad():
        # 사전 로딩된 test_data 리스트에서 이미지 정보(item)를 하나씩 꺼내와 분석합니다.
        for item in tqdm(test_data, desc="Analyzing test set"):
            # 딕셔너리 형태의 item에서 'full_image' 키를 이용해 NumPy 배열 형태의 원본 이미지를 가져옵니다.
            image_np = item["full_image"]

            x, y, w, h = ROI_RECT
            # 원본 이미지(image_np)에서 해당 좌표와 크기만큼을 잘라내어 관심 영역(ROI) 이미지를 만듭니다.
            roi_image_np = image_np[y:y + h, x:x + w]

            # albumentations로 정의된 변환(transform_roi)을 적용하여 ROI 이미지를 모델 입력 형식에 맞게 처리합니다.
            # (예: 이미지 리사이즈, 정규화 등)
            transformed = transform_roi(image=roi_image_np)

            # 변환된 이미지를 PyTorch 텐서로 만들고, 모델이 배치(batch) 단위로 입력을 받기 때문에 unsqueeze(0)으로 배치 차원을 추가합니다.
            # .to(device)를 통해 텐서를 지정된 장치(GPU 또는 CPU)로 보냅니다.
            roi_tensor = transformed["image"].unsqueeze(0).to(device)

            # 준비된 ROI 텐서를 모델에 입력하여 이상 탐지 분석을 수행하고, 그 결과를 outputs 변수에 저장합니다.
            outputs = model(roi_tensor)
            # 모델 출력(outputs)에서 이미지 전체의 이상 점수(anomaly score)를 가져옵니다. .item()은 텐서에서 숫자 값만 추출하는 함수입니다.
            pred_score = outputs[0].item()
            # 모델 출력에서 픽셀별 이상 점수를 담고 있는 이상 맵(anomaly map)을 가져옵니다.
            # .squeeze()로 불필요한 차원을 제거하고, 시각화를 위해 .cpu().numpy()로 NumPy 배열로 변환합니다.
            anomaly_map = outputs[2].squeeze().cpu().numpy()
            # 계산된 이상 점수를 미리 설정한 임계값(ANOMALY_THRESHOLD)과 비교하여 최종 판정(0: 정상, 1: 불량)을 내립니다.
            pred_label = 1 if pred_score > ANOMALY_THRESHOLD else 0

            # anomaly_map에서 가장 높은 값을 갖는 픽셀의 위치(y, x 인덱스)를 찾습니다.
            max_loc_y, max_loc_x = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
            # 해당 위치의 실제 픽셀 값(가장 높은 이상치)을 저장합니다.
            max_anomaly_value = np.max(anomaly_map)

            # anomaly_map의 좌표는 리사이즈된 이미지 기준이므로, 원본 ROI 좌표계로 변환합니다.
            h_map, w_map = anomaly_map.shape
            # (예: 256x256 -> 881x324)
            loc_in_roi_x = int(max_loc_x * w / w_map)
            loc_in_roi_y = int(max_loc_y * h / h_map)


            # 나중에 종합 리포트를 생성하기 위해 현재 이미지의 분석 결과(경로, 실제 라벨, 예측 라벨 등)를 리스트에 저장합니다.
            results.append({
                "path": item["path"], "gt_label": item["gt_label"], "pred_label": pred_label,
                "pred_score": pred_score, "gt_defect": item["gt_defect"],
                "max_anomaly_loc_in_roi": (loc_in_roi_x, loc_in_roi_y),  # (x, y) 순서로 저장
                "max_anomaly_value": max_anomaly_value
            })

            # --- 여기부터는 결과 시각화(히트맵 생성 및 이미지 저장) 과정입니다. ---

            # 이상 맵(anomaly_map)의 값 범위를 0과 1 사이로 정규화하여 시각적으로 표현하기 쉽게 만듭니다. (1e-6은 0으로 나누는 오류 방지용)
            heatmap = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map) + 1e-6)
            # 0~1 범위의 히트맵을 0~255 범위의 8비트 정수형(이미지 형식)으로 변환합니다.
            heatmap = (heatmap * 255).astype(np.uint8)
            # 흑백 히트맵에 컬러맵(JET)을 적용하여 이상 점수가 높은 영역(붉은색)과 낮은 영역(푸른색)을 쉽게 구분하도록 합니다.
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # 생성된 컬러 히트맵을 원본 ROI의 너비(w)와 높이(h)로 다시 조정합니다.
            heatmap_resized = cv2.resize(heatmap_color, (w, h))

            # OpenCV는 BGR 색상 순서를 사용하므로, 시각화를 위해 원본 이미지(RGB)를 BGR로 변환합니다.
            original_image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            # 원본 이미지와 동일한 크기의 검은색 빈 이미지(도화지)를 생성합니다.
            full_heatmap_overlay = np.zeros_like(original_image_bgr)
            # 빈 이미지의 ROI 위치에 리사이즈된 컬러 히트맵을 붙여넣습니다.
            full_heatmap_overlay[y:y + h, x:x + w] = heatmap_resized

            # 히트맵의 투명도를 설정합니다 (0.5는 반투명을 의미).
            alpha = 0.5
            # 원본 이미지와 히트맵 오버레이 이미지를 위에서 설정한 투명도를 적용하여 자연스럽게 합성합니다.
            visualization = cv2.addWeighted(original_image_bgr, 1 - alpha, full_heatmap_overlay, alpha, 0)

            # 예측 라벨(pred_label)이 1(불량)이면 빨간색(0,0,255), 0(정상)이면 초록색(0,255,0)으로 사각형 색상을 결정합니다.
            roi_color = (0, 0, 255) if pred_label == 1 else (0, 255, 0)
            # 합성된 이미지의 ROI 영역에 위에서 결정한 색상으로 사각형을 그립니다.
            cv2.rectangle(visualization, (x, y), (x + w, y + h), roi_color, 2)

            # 이미지에 표시할 텍스트(실제 라벨, 예측 결과, 이상 점수)를 생성합니다.
            text = f"GT: {item['gt_defect']} | Pred: {'Anomaly' if pred_label else 'Normal'} ({pred_score:.2f})"
            # 생성한 텍스트를 이미지의 ROI 사각형 위쪽에 흰색으로 추가합니다.
            cv2.putText(visualization, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 완성된 시각화 이미지를 지정된 결과 폴더(RESULTS_SAVE_DIR)에 원본 파일명과 동일한 이름으로 저장합니다.
            cv2.imwrite(str(RESULTS_SAVE_DIR / item["path"].name), visualization)

    # --- 요청하신 종합 리포트 출력 ---
    gt_labels = [r['gt_label'] for r in results]
    pred_labels = [r['pred_label'] for r in results]
    pred_scores = [r['pred_score'] for r in results]

    auroc_score = "N/A"
    if len(set(gt_labels)) > 1:
        auroc_score = f"{roc_auc_score(gt_labels, pred_scores):.4f}"
    else:
        print("[!] 경고: 정답 라벨에 클래스가 하나뿐이므로 AUROC를 계산할 수 없습니다.")

    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels, labels=[0, 1]).ravel()

    print("\n" + "=" * 50)
    print("종합 성능 리포트")
    print("=" * 50)
    print(f"사용된 이상 점수 임계값: {ANOMALY_THRESHOLD}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  - TN (정상->정상): {tn}")
    print(f"  - FP (정상->불량): {fp}  <-- 오탐지")
    print(f"  - TP (불량->불량): {tp}")
    print(f"  - FN (불량->정상): {fn}  <-- 미탐지")
    print("-" * 50)
    print("주요 성능 지표:")
    print(f"  - Accuracy: {accuracy_score(gt_labels, pred_labels):.4f}")
    print(f"  - F1-Score: {f1_score(gt_labels, pred_labels):.4f}")
    print(f"  - AUROC: {auroc_score}")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("상세 분석 목록")
    print("=" * 50)
    fp_list = [r for r in results if r['gt_label'] == 0 and r['pred_label'] == 1]
    fn_list = [r for r in results if r['gt_label'] == 1 and r['pred_label'] == 0]
    print(f"\n[오탐지된 정상 이미지 (False Positives) - 총 {len(fp_list)}개]")
    for r in fp_list: print(f"  - {r['path'].name} (Score: {r['pred_score']:.2f})")
    print(f"\n[미탐지된 불량 이미지 (False Negatives) - 총 {len(fn_list)}개]")
    for r in fn_list: print(f"  - {r['path'].name} (GT: {r['gt_defect']}, Score: {r['pred_score']:.2f})")
    print("\n[정상 탐지된 불량 이미지 (True Positives)]")
    tp_list = [r for r in results if r['gt_label'] == 1 and r['pred_label'] == 1]
    for r in tp_list: print(f"  - {r['path'].name} (GT: {r['gt_defect']}, Score: {r['pred_score']:.2f})")
    print("=" * 50)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    if FORCE_RETRAIN or not MODEL_SAVE_PATH.exists():
        train_model(device)  # <--- 여기서 학습 함수를 호출합니다.
    else:
        print("FORCE_RETRAIN=False이고 모델 파일이 이미 존재하므로, 학습을 건너뛰고 바로 분석을 시작합니다.")

    analyze_results(device)