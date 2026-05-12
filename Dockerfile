# Project-specific image for robust-meme-hate-detection.
# Layered on top of the EE-559 course base image, which already provides
# CUDA-enabled PyTorch. We add open_clip_torch and the rest of the project
# requirements, then pre-bake CLIP weights so compute pods without outbound
# internet still start cleanly.

FROM registry.rcp.epfl.ch/ee559/environment-with-packages:latest

USER root

# Run:AI launches with --run-as-uid 316498. Ensure the user exists in
# /etc/passwd so getpwuid() (used by torch's cache-dir setup, getpass.getuser,
# etc.) can resolve it. Setting USER/LOGNAME/HOME also short-circuits the
# pwd lookup in most code paths.
RUN id 316498 >/dev/null 2>&1 || \
    echo "guasch:x:316498:0:guasch:/home/guasch:/bin/bash" >> /etc/passwd && \
    mkdir -p /home/guasch && chown 316498:0 /home/guasch
ENV HOME=/home/guasch USER=guasch LOGNAME=guasch

WORKDIR /tmp/build
COPY requirements.txt /tmp/build/requirements.txt
# The base image's Python is PEP-668 "externally managed". This is a
# course-controlled image and our additions are layered on top, so
# --break-system-packages is acceptable here.
RUN pip install --no-cache-dir --break-system-packages -r /tmp/build/requirements.txt

# Pre-bake the OpenCLIP weights for ViT-B/32 (and ViT-L/14 as the documented
# first ablation). Failing to download falls back without breaking the build,
# but the warning will appear in the build log.
RUN python3 - <<'PY' || true
import open_clip
for arch, pretrained in [("ViT-B-32", "laion2b_s34b_b79k"), ("ViT-L-14", "laion2b_s32b_b82k")]:
    try:
        open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        print(f"baked {arch}/{pretrained}")
    except Exception as exc:
        print(f"WARN: could not pre-download {arch}/{pretrained}: {exc}")
PY

# The container runs jobs as the EPFL UID; the base image ships a /home/guasch
# already (uid 316498). No USER switch needed because Run:AI submits with
# --run-as-uid 316498.
