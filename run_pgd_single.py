"""Run PGD on a single image and save the adversarial result."""
import sys
sys.path.insert(0, "src")

import torch
from PIL import Image
from robust_meme_hate_detection.attacks.pgd import pgd_image
from robust_meme_hate_detection.data.transforms import ClipTokenize
from torchvision import transforms as T
from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier

import os
CKPT = os.environ.get(
    "CKPT",
    "/scratch/robust-meme-hate-detection/experiments/stage1-seed0-20260503-163917/ckpt/best.pt",
)
IMAGE = "/scratch/datasets/hate_meta/img/01456.png"
TEXT  = "they see them rollin..... they hating.."
LABEL = 1
EPSILON = 12 / 255.0
ALPHA   = 0.25 * EPSILON
STEPS   = 50
OUT     = "/scratch/datasets/perturbed/01456.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load(CKPT, map_location=device, weights_only=False)
model = CLIPHateMemeClassifier(
    arch="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    freeze_encoders=True,
    head_hidden=512,
    head_dropout=0.2,
).to(device)
model.load_state_dict(state["model_state"])
model.eval()

img_tfm = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
])
tok_tfm = ClipTokenize(arch="ViT-B-32")

image   = img_tfm(Image.open(IMAGE).convert("RGB")).unsqueeze(0).to(device)
tokens  = tok_tfm(TEXT).unsqueeze(0).to(device)
label_t = torch.tensor([LABEL], dtype=torch.float32, device=device)

with torch.no_grad():
    clean_prob = torch.sigmoid(model(image, tokens)).item()

adv = pgd_image(model, image, tokens, label_t, epsilon=EPSILON, alpha=ALPHA, steps=STEPS)

with torch.no_grad():
    adv_prob = torch.sigmoid(model(adv, tokens)).item()

adv_img = (adv[0].cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
Image.fromarray(adv_img).save(OUT)

print(f"clean prob: {clean_prob:.4f}  ->  adv prob: {adv_prob:.4f}")
print(f"saved: {OUT}")
