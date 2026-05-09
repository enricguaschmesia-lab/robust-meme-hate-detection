"""Example usage of perturbations with the Hateful Memes dataset."""

import json
import os
from pathlib import Path

# Detect if running on cluster or locally
DATA_ROOT = os.environ.get('DATA_ROOT', 'data/raw/data')
# Save perturbed dataset in the same location as the original data
OUTPUT_ROOT = Path(DATA_ROOT).parent / 'perturbed'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Example 1: Clean data (no perturbations)
from robust_meme_hate_detection.data.hateful_memes import build_hateful_memes_dataloader

clean_loader = build_hateful_memes_dataloader(
    dataset_root=DATA_ROOT,
    split='dev',
    batch_size=4,
    apply_perturbations=False,
)

# Example 2: Text-only perturbations
from robust_meme_hate_detection.perturbations import TextPerturbation, ComposePerturbation

text_pipeline = ComposePerturbation([
    TextPerturbation(mode='leetspeak', probability=0.3, severity=0.5),
    TextPerturbation(mode='char_deletion', probability=0.2, severity=0.1),
    TextPerturbation(mode='spacing', probability=0.2, severity=0.3),
])

text_only_loader = build_hateful_memes_dataloader(
    dataset_root=DATA_ROOT,
    split='dev',
    batch_size=4,
    apply_perturbations=True,
    text_perturbation=text_pipeline,
)

# Example 3: Image-only perturbations
from robust_meme_hate_detection.perturbations import ImagePerturbation

image_pipeline = ComposePerturbation([
    ImagePerturbation(mode='gaussian_noise', probability=0.4, severity=0.05),
    ImagePerturbation(mode='blur', probability=0.3, severity=0.3),
    ImagePerturbation(mode='brightness', probability=0.2, severity=0.2),
])

image_only_loader = build_hateful_memes_dataloader(
    dataset_root=DATA_ROOT,
    split='dev',
    batch_size=4,
    apply_perturbations=True,
    image_perturbation=image_pipeline,
)

# Example 4: Combined multimodal perturbations (both text and image)
combined_loader = build_hateful_memes_dataloader(
    dataset_root=DATA_ROOT,
    split='dev',
    batch_size=4,
    apply_perturbations=True,
    text_perturbation=text_pipeline,
    image_perturbation=image_pipeline,
)

# Example 5: Guaranteed both modalities perturbed
guaranteed_both = ComposePerturbation([
    TextPerturbation(mode='leetspeak', probability=1.0, severity=0.5),
    TextPerturbation(mode='char_deletion', probability=1.0, severity=0.1),
])

guaranteed_image = ComposePerturbation([
    ImagePerturbation(mode='gaussian_noise', probability=1.0, severity=0.05),
    ImagePerturbation(mode='blur', probability=1.0, severity=0.3),
])

robust_loader = build_hateful_memes_dataloader(
    dataset_root=DATA_ROOT,
    split='dev',
    batch_size=4,
    apply_perturbations=True,
    text_perturbation=guaranteed_both,
    image_perturbation=guaranteed_image,
)

if __name__ == '__main__':
    loaders = [
        ('clean', clean_loader),
        ('text-only', text_only_loader),
        ('image-only', image_only_loader),
        ('combined', combined_loader),
        ('guaranteed-both', robust_loader),
    ]
    
    for name, loader in loaders:
        print(f"\nProcessing {name}...")
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(parents=True, exist_ok=True)
        img_dir = output_dir / 'img'
        img_dir.mkdir(exist_ok=True)
        
        metadata = []
        sample_id = 0
        
        # Iterate through dataset to save perturbed samples
        for batch in loader:
            for i in range(len(batch['image'])):
                image = batch['image'][i]
                text = batch['text'][i]
                label = batch['label'][i].item()
                original_id = batch['id'][i]
                
                # Save image
                img_path = f"img/{sample_id:05d}.png"
                image.save(output_dir / img_path)
                
                # Collect metadata
                metadata.append({
                    'id': str(sample_id).zfill(5),
                    'img': img_path,
                    'text': text,
                    'label': label,
                    'original_id': original_id,
                })
                sample_id += 1
        
        # Save metadata JSONL
        jsonl_path = output_dir / f'{name}.jsonl'
        with jsonl_path.open('w') as f:
            for item in metadata:
                f.write(json.dumps(item) + '\n')
        
        print(f"  Saved {sample_id} samples to {output_dir}")
        print(f"  JSONL: {jsonl_path}")
    
    print(f"\n✓ All perturbed datasets saved to {OUTPUT_ROOT}")
