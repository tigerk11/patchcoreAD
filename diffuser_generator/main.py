import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import random

# 1. 수정한 데이터 로더 임포트
from mvtec_dataset import MVTecDataset

# 2. Hugging Face 라이브러리 임포트
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import \
    StableDiffusionInpaintPipeline as StableDiffusionInpaintingPipeline


def create_random_mask(image_size):
    """
    이미지 위에 무작위 사각형 마스크를 생성합니다.
    """
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    mask_w = random.randint(int(width * 0.1), int(width * 0.3))
    mask_h = random.randint(int(height * 0.1), int(height * 0.3))
    mask_x = random.randint(0, width - mask_w)
    mask_y = random.randint(0, height - mask_h)
    draw.rectangle([mask_x, mask_y, mask_x + mask_w, mask_y + mask_h], fill=255)
    return mask


def main():
    # --- 설정 ---
    ROOT_DIR = 'C:/Users/AI-00/Desktop/mvtec_anomaly_detection'
    CATEGORY = 'capsule'
    SAVE_DIR = f'./results/{CATEGORY}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. 데이터셋 준비 ---
    img_size = 512
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # 'train' phase의 데이터만 불러옵니다. (데이터 누수 방지)
    print("Loading 'good' images from the train set for defect generation...")
    # 'train' phase는 개선된 MVTecDataset 클래스에서 자동으로 'good' 폴더만 로드합니다.
    train_dataset = MVTecDataset(root_dir=ROOT_DIR, category=CATEGORY, phase='train', transform=transform)

    # 'train' 데이터셋만을 사용하여 DataLoader를 생성합니다.
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(f"Total {len(train_dataset)} normal images loaded for generation.")

    # --- 2. 확산 모델(Inpainting) 로드 ---
    pipe = StableDiffusionInpaintingPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to(device)

    to_pil = transforms.ToPILImage()

    # --- 이미지 생성 루프 ---
    generated_count = 0
    target_count = 1

    while generated_count < target_count:
        # dataloader가 이제 (image, label, path) 3개를 반환합니다.
        for image_tensor, label, img_path in dataloader:
            if generated_count >= target_count:
                break

            pil_image = to_pil(image_tensor.squeeze(0))
            mask_image = create_random_mask(pil_image.size)
            # cable
            # prompts
            prompts = [
                "crack, faulty_imprint, poke",
                "rotated, scratch, squeeze"
            ]
            prompt = random.choice(prompts)

            print(
                f"Processing image {generated_count + 1}/{target_count} from '{os.path.basename(img_path[0])}' with prompt: '{prompt}'")

            generated_image = pipe(
                prompt=prompt,
                image=pil_image,
                mask_image=mask_image,
                strength=0.3
            ).images[0]

            save_path_generated = os.path.join(SAVE_DIR, f"generated_{generated_count:04d}.png")
            generated_image.save(save_path_generated)

            generated_count += 1

    print(f"Finished generating {generated_count} images.")


if __name__ == '__main__':
    main()