import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, phase='train', transform=None):
        """
        MVTec 데이터셋을 정상/불량 라벨과 함께 불러오는 최종 클래스

        :param root_dir: mvtec_anomaly_detection 데이터의 기본 경로
        :param category: 'bottle', 'capsule' 등 특정 품목 폴더명
        :param phase: 'train' or 'test' (학습 또는 테스트 데이터)
        :param transform: 이미지 변환을 적용할 torchvision transforms
        """
        self.root_dir = os.path.join(root_dir, category)
        self.phase = phase
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 이미지에 해당하는 라벨(0:정상, 1:불량)을 저장할 리스트
        self.mask_paths = []  # 마스크 경로를 저장할 리스트 추가

        # 데이터 로드 로직
        self._load_dataset()

    def _load_dataset(self):
        """데이터 경로와 라벨을 로드하는 내부 함수"""
        if self.phase == 'train':
            # 학습 데이터는 'good' 폴더의 정상 이미지만 사용
            train_dir = os.path.join(self.root_dir, 'train', 'good')
            image_fnames = self._get_image_fnames(train_dir)
            self.image_paths.extend([os.path.join(train_dir, fname) for fname in image_fnames])
            self.labels.extend([0] * len(image_fnames))  # 정상 라벨(0) 추가
            self.mask_paths.extend([""] * len(image_fnames))  # 정상 이미지는 마스크 경로가 없음

        elif self.phase == 'test':
            # 테스트 데이터는 'good' 폴더와 모든 불량 폴더를 사용
            test_dir = os.path.join(self.root_dir, 'test')
            defect_types = sorted(os.listdir(test_dir))

            for defect_type in defect_types:
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir): continue

                image_fnames = self._get_image_fnames(defect_dir)
                image_paths_for_type = [os.path.join(defect_dir, fname) for fname in image_fnames]
                self.image_paths.extend(image_paths_for_type)

                # 'good' 폴더는 정상(0), 나머지는 모두 불량(1)으로 라벨링
                if defect_type == 'good':
                    self.labels.extend([0] * len(image_fnames))
                    self.mask_paths.extend([""] * len(image_fnames))
                else:
                    self.labels.extend([1] * len(image_fnames))
                    # 불량 이미지에 해당하는 마스크 경로를 생성하여 추가
                    mask_fnames = [f"{Path(fname).stem}_mask.png" for fname in image_fnames]
                    mask_paths_for_type = [os.path.join(self.root_dir, 'ground_truth', defect_type, fname) for fname in
                                           mask_fnames]
                    self.mask_paths.extend(mask_paths_for_type)

        else:
            raise ValueError(f"Invalid phase: {self.phase}. Choose 'train' or 'test'.")

    def _get_image_fnames(self, directory):
        """주어진 디렉토리에서 이미지 파일 이름 목록을 반환"""
        if not os.path.isdir(directory): return []
        return sorted([fname for fname in os.listdir(directory) if fname.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        """데이터셋의 전체 이미지 개수를 반환"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 이미지, 라벨, 경로를 반환
        """
        """
                주어진 인덱스에 해당하는 데이터를 딕셔너리 형태로 반환
                """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if label == 0:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.array(Image.open(mask_path).convert("L"))

        # 이 부분은 albumentations 변환기를 기대합니다 (image=, mask= 사용)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {
            "image": image,
            "label": label,
            "mask": mask.unsqueeze(0),
            "image_path": img_path
        }