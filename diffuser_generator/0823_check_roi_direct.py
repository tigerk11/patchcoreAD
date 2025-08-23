# check_roi_direct.py

import cv2
from pathlib import Path
import numpy as np

# 제가 추천드렸던, 캡슐 표면을 정확히 가리키는 ROI 좌표입니다.
ROI_RECT = (67, 330, 881, 324)

# 실제 존재하는 정상 이미지 파일 경로
IMAGE_PATH = Path("C:/Users/AI-00/Desktop/capsule/train/good/000.png")


def final_check():
    print("=" * 60)
    print("--- 최종 원인 규명 테스트 시작 ---")
    print(f"테스트 대상 파일: {IMAGE_PATH}")
    print(f"테스트 ROI 좌표: {ROI_RECT}")
    print("=" * 60)

    if not IMAGE_PATH.is_file():
        print(f"[!!!] 오류: 해당 경로에 파일이 없습니다: {IMAGE_PATH}")
        return

    # 1. 오직 OpenCV로만 이미지를 불러옵니다.
    original_image = cv2.imread(str(IMAGE_PATH))

    if original_image is None:
        print("[!!!] 오류: cv2.imread가 이미지를 불러오지 못했습니다. 파일이 손상되었을 수 있습니다.")
        return

    print("이미지를 성공적으로 불러왔습니다.")
    print(f"불러온 원본 이미지 크기: {original_image.shape}")

    # 2. 불러온 이미지에서 ROI 영역을 잘라냅니다.
    x, y, w, h = ROI_RECT
    roi_crop = original_image[y:y + h, x:x + w]
    print(f"ROI 영역을 성공적으로 잘라냈습니다. 잘라낸 크기: {roi_crop.shape}")

    # 3. 잘라낸 ROI를 파일로 저장합니다.
    SAVE_PATH = Path("./direct_roi_check.png")
    cv2.imwrite(str(SAVE_PATH), roi_crop)
    print(f"\n결과를 '{SAVE_PATH}' 파일로 저장했습니다.")
    print("--- 테스트 종료 ---")
    print(f"\n이제 프로젝트 폴더의 '{SAVE_PATH}' 파일을 열어서")
    print("이미지가 검은색인지, 아니면 캡슐 표면인지 확인해주세요.")


if __name__ == "__main__":
    final_check()