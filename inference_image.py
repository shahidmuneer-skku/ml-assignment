import argparse
import os
import glob
import cv2
import numpy as np
import torch
from typing import List, Tuple



# If your model path differs, adjust the import below
from models.alignment_pretrained.model_with_blip import MMModerator
from PIL import Image

from models.CLIPSvD import CLIPSvD

import torchvision.transforms.functional as TF
import torch.nn.functional as F

from models.alignment_pretrained.unet import UNetImageDecoder
from pathlib import Path
yunet = cv2.FaceDetectorYN.create(
    model='/media/NAS/USERS/shahid/MultimodalAudioVisualModerator/third_party/yunet/face_detection_yunet_2023mar.onnx',  # download from OpenCV zoo
    config='',
    input_size=(320, 320),  # small size = fast
    score_threshold=0.5, nms_threshold=0.3, top_k=5000
)
def detect_faces_yunet( img_any):
    """
    yunet: cv2.FaceDetectorYN instance
    img_any: torch tensor / PIL / np
    Returns: list of (x1, y1, x2, y2, score)
    """
    bgr = as_bgr_uint8_np(img_any)
    h, w = bgr.shape[:2]
    yunet.setInputSize((w, h))

    # OpenCV 4.x returns either dets or (ok, dets) depending on build – handle both:
    out = yunet.detect(bgr)
    dets = out[1] if isinstance(out, tuple) else out

    boxes = []
    if dets is not None and len(dets) > 0:
        dets = np.array(dets)
        # YuNet rows: [x, y, w, h, score, ... 10 landmark values]
        for d in dets:
            x, y, ww, hh, s = d[:5]
            x1, y1, x2, y2 = int(x), int(y), int(x + ww), int(y + hh)
            boxes.append((x1, y1, x2, y2, float(s)))
    return boxes

# def to_pil(x):
#     # accept torch or numpy
#     if isinstance(x, torch.Tensor):
#         arr = x.detach().cpu().numpy()
#     elif isinstance(x, np.ndarray):
#         arr = x
#     else:
#         raise TypeError(f"to_pil: unsupported type {type(x)}")

#     # drop batch
#     if arr.ndim == 4:
#         arr = arr[0]

#     # CHW -> HWC
#     if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
#         arr = np.transpose(arr, (1, 2, 0))

#     # map to uint8 if needed
#     if arr.dtype != np.uint8:
#         arr = np.clip(arr, 0, 255).astype(np.uint8)

#     # handle shapes
#     if arr.ndim == 2:
#         return Image.fromarray(arr, mode="L")
#     if arr.ndim == 3:
#         if arr.shape[2] == 1:
#             # either grayscale L...
#             # return Image.fromarray(arr.squeeze(-1), mode="L")
#             # ...or expand to RGB to keep downstream code consistent:
#             arr = np.repeat(arr, 3, axis=2)
#             return Image.fromarray(arr, mode="RGB")
#         if arr.shape[2] in (3, 4):
#             return Image.fromarray(arr)  # RGB/RGBA
#     raise ValueError(f"to_pil: unexpected array shape {arr.shape}")


def to_pil(tensor):
    # Handle [B,C,H,W] or [C,H,W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)   # drop batch
    if tensor.dtype != torch.uint8:
        tensor = (tensor.clamp(0,1) * 255).to(torch.uint8)
    return TF.to_pil_image(tensor)



# ==========================
# Model loading
# ==========================

def create_vision_encoder():

    
            
    REPO_DIR = "/media/NAS/USERS/shahid/MultimodalAudioVisualModerator/dinov3"
    # # # dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights="https://dinov3.llamameta.net/dinov3_convnext_large/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYWUwNXlycnV1NjE3cW96dXJubGh3bnkxIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTg5NDI3NDB9fX1dfQ__&Signature=EryhpWx6QmuBLIk3kLu-hITTm2hEN72AS3cINgccRB25fSZbg7AN-dw14OHiQr8yH--vCYpSOPGf5ar3ZC7%7EwFZLlGuf799u5ULxucKLj35H%7EEh9ZgBmX8IRhqIwzTldRx3npPtyRd19roSebbHAElWUSaiHfuVmN-gKTdyRHslGsG9tPOlRTndKib%7EKX-40s4IqJAs%7Eyg-okLdWWA5DevBz5oSX4o0zrf0DxwaP4CvU4c9VHH6pSUG2ga9PaMoOLLk4J8ero5FQlxihBSzKHJCpMRfpMUNjtH055fDPNBYnpHhMeY6iH1XBDVzPP1QHcXP1MwZbN8PvMROin24-IQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=781072348038381")
    
    # # # self.encoder = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights="https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieWJhOTd6MTQ1Yjl6Mmx5ZnN4a2V6ZWFpIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTk0ODExMjd9fX1dfQ__&Signature=bwYAHsGsZoKD4l5nV%7ECINVD9Ad50wNW9SXLYst9hkeA1TUrpRhKLvH3j8jpDQCzZ7HrTQBokpUV15pVUhdPwPGj4FpV0VbSn-Pk675P6AquCe94%7E-z9VA7WP-ZdeaApZag7DoZaTyAbuFk27Yb0TX0aHRzfYDVge%7EKmSEHwgs1lsBxUnXRX%7Ebl9t7xwQl%7EFWcoDGsE85GtwC6PIDsGybm1E3qHvtkE2KdN1PUoQGswYU0JzxVFkRjfrMjYz2AbMhEnSP47HOYh1stwoTyoRzC-U5RjK7w0cMvMW6Y4nMqMz7MYf9H5Uf%7EAbJZrE5DnimJorQ3WHuZ1KPBHapfGHB1g__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2010796719734569")
        
        
    # # # self.encoder = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights="https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieWJhOTd6MTQ1Yjl6Mmx5ZnN4a2V6ZWFpIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTk0ODExMjd9fX1dfQ__&Signature=bwYAHsGsZoKD4l5nV%7ECINVD9Ad50wNW9SXLYst9hkeA1TUrpRhKLvH3j8jpDQCzZ7HrTQBokpUV15pVUhdPwPGj4FpV0VbSn-Pk675P6AquCe94%7E-z9VA7WP-ZdeaApZag7DoZaTyAbuFk27Yb0TX0aHRzfYDVge%7EKmSEHwgs1lsBxUnXRX%7Ebl9t7xwQl%7EFWcoDGsE85GtwC6PIDsGybm1E3qHvtkE2KdN1PUoQGswYU0JzxVFkRjfrMjYz2AbMhEnSP47HOYh1stwoTyoRzC-U5RjK7w0cMvMW6Y4nMqMz7MYf9H5Uf%7EAbJZrE5DnimJorQ3WHuZ1KPBHapfGHB1g__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2010796719734569")
        
    # # # self.encoder = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights="https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoib3VpbXR2cHlhZXE5c2JwajNucnN3aWF2IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTk5MDI0NDF9fX1dfQ__&Signature=GIYJR4%7ESJVx0gkcm7lgAvDljIfpR30MXgWb2VpCqbDeVpnwjn97k%7EOcPeGF-lkR0q1Sn3Iw5Y3iYWqspcpPoDJ4FXUmMKhWtd-m00HO73Aknq2kyrKVMBpzwQB-k-2zZe7okJfXTj46EWbzu9mNcxt%7ErDPe7phQpRJi8Dleida1BJ823oXFx8d7oRSa4NDSzT2TNXqNNZ8ux7N0aDfdT9dupEeEr4AP06LhYB2I7kF%7Ef4bvKQsKnlPMVDAADyYG9nQ7HqAW41LaWZtR-BrDGm%7ESNu-6L44cUVnk3qEPVRMQB4GW7ixRGGhtr37F6HVz%7EKilrCpivFD6ej4reNUWaGQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=829796286371500")
        
    # encoder = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights="https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoib3VpbXR2cHlhZXE5c2JwajNucnN3aWF2IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTk5MDI0NDF9fX1dfQ__&Signature=GIYJR4%7ESJVx0gkcm7lgAvDljIfpR30MXgWb2VpCqbDeVpnwjn97k%7EOcPeGF-lkR0q1Sn3Iw5Y3iYWqspcpPoDJ4FXUmMKhWtd-m00HO73Aknq2kyrKVMBpzwQB-k-2zZe7okJfXTj46EWbzu9mNcxt%7ErDPe7phQpRJi8Dleida1BJ823oXFx8d7oRSa4NDSzT2TNXqNNZ8ux7N0aDfdT9dupEeEr4AP06LhYB2I7kF%7Ef4bvKQsKnlPMVDAADyYG9nQ7HqAW41LaWZtR-BrDGm%7ESNu-6L44cUVnk3qEPVRMQB4GW7ixRGGhtr37F6HVz%7EKilrCpivFD6ej4reNUWaGQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=829796286371500")
        
    # encoder = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights="https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoib3VpbXR2cHlhZXE5c2JwajNucnN3aWF2IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTk5MDI0NDF9fX1dfQ__&Signature=GIYJR4%7ESJVx0gkcm7lgAvDljIfpR30MXgWb2VpCqbDeVpnwjn97k%7EOcPeGF-lkR0q1Sn3Iw5Y3iYWqspcpPoDJ4FXUmMKhWtd-m00HO73Aknq2kyrKVMBpzwQB-k-2zZe7okJfXTj46EWbzu9mNcxt%7ErDPe7phQpRJi8Dleida1BJ823oXFx8d7oRSa4NDSzT2TNXqNNZ8ux7N0aDfdT9dupEeEr4AP06LhYB2I7kF%7Ef4bvKQsKnlPMVDAADyYG9nQ7HqAW41LaWZtR-BrDGm%7ESNu-6L44cUVnk3qEPVRMQB4GW7ixRGGhtr37F6HVz%7EKilrCpivFD6ej4reNUWaGQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=829796286371500")
        
    # # # encoder = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights="https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoib3VpbXR2cHlhZXE5c2JwajNucnN3aWF2IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTk5MDI0NDF9fX1dfQ__&Signature=GIYJR4%7ESJVx0gkcm7lgAvDljIfpR30MXgWb2VpCqbDeVpnwjn97k%7EOcPeGF-lkR0q1Sn3Iw5Y3iYWqspcpPoDJ4FXUmMKhWtd-m00HO73Aknq2kyrKVMBpzwQB-k-2zZe7okJfXTj46EWbzu9mNcxt%7ErDPe7phQpRJi8Dleida1BJ823oXFx8d7oRSa4NDSzT2TNXqNNZ8ux7N0aDfdT9dupEeEr4AP06LhYB2I7kF%7Ef4bvKQsKnlPMVDAADyYG9nQ7HqAW41LaWZtR-BrDGm%7ESNu-6L44cUVnk3qEPVRMQB4GW7ixRGGhtr37F6HVz%7EKilrCpivFD6ej4reNUWaGQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=829796286371500")
            
            
    # for p in encoder.parameters():
    #         p.requires_grad = False

    # for i, block in enumerate(encoder.blocks):
    #     if i < 3:  # unfreeze first 3 blocks
    #         for p in block.parameters():
    #             p.requires_grad = True


    # model = DINOEncoderLoRA(encoder, r=16, emb_dim=1024, use_lora=True)
    


            
    # # Optionally: also allow qkv projection (but only small LoRA rank adapters)
    # for name, module in encoder.named_modules():
    #     if "qkv" in name:   # some ViT implementations call it 'attn.qkv'
    #         for p in module.parameters():
    #             p.requires_grad = True
    #         print(f"Unfroze QKV: {name}")
# 
    # return encoder
    model = CLIPSvD()

    return model
   


def load_model(checkpoint_path: str, device: torch.device) -> MMModerator:
    """Load the MMModerator model from a checkpoint and set to eval mode."""
    raw = torch.load(checkpoint_path, map_location=device)
    # If you saved a dict with optimizer, etc., prefer the model_state_dict key when present
    sd = raw.get("model_state_dict", raw)
    
    # Strip any DistributedDataParallel prefixes like "module."
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    
    vision_encoder = create_vision_encoder()
    unet_decoder = UNetImageDecoder(
            num_patches=256,       # 7 × 7 grid (ViT-B/32)
            token_dim=1024,        # ViT-B/32 embedding dim
            out_channels=3,       # mask or 3 for RGB
            base_channels=256,
            img_size=256,
            grid_hw=(16, 16)        # explicitly set to match patch grid
        )
        
    checkpoint_path_encoder = os.path.join(os.path.dirname(checkpoint_path), "model_state_encoder.pt")
    checkpoint_path_decoder = os.path.join(os.path.dirname(checkpoint_path), "model_state_decoder.pt")
    raw_encoder   = torch.load(checkpoint_path_encoder, map_location=device)
    raw_decoder   = torch.load(checkpoint_path_decoder, map_location=device)
    
    sd_encoder    = raw_encoder.get("model_state_dict", raw_encoder)

    # strip off any "module." prefixes
    new_sd_encoder = {}
    for k, v in sd_encoder.items():
        new_key = k.replace("module.", "")  
        new_sd_encoder[new_key] = v
        
    vision_encoder.load_state_dict(new_sd_encoder)        # strict=True by default
    

    sd_decoder    = raw_decoder.get("model_state_dict", raw_decoder)

    # strip off any "module." prefixes
    new_sd_decoder = {}
    for k, v in sd_decoder.items():
        new_key = k.replace("module.", "")  
        new_sd_decoder[new_key] = v
        
    unet_decoder.load_state_dict(new_sd_decoder)        # strict=True by default
    
    model = MMModerator(pretraining=False,vision_encoder=vision_encoder, unet_decoder=unet_decoder, num_classes=1)
    model.load_state_dict(new_sd)
    model.to(device).eval()
    return model

# ==========================
# Image loading / preprocessing
# ==========================

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel RGB uint8."""
    if img is None:
        raise ValueError("Failed to read image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _equalize_y_channel(rgb: np.ndarray) -> np.ndarray:
    """Histogram equalize the Y channel in YCrCb space, return RGB uint8."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _center_crop_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        m = (h - w) // 2
        return img[m:m + w, :]
    else:
        m = (w - h) // 2
        return img[:, m:m + h]

# CLIP mean and std (ImageNet normalization)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def unnormalize(t: torch.Tensor, mean=CLIP_MEAN, std=CLIP_STD):
    """
    Reverse CLIP normalization → return tensor in [0,1].
    Supports [C,H,W] and [B,C,H,W].
    """
    mean = torch.tensor(mean, device=t.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=t.device).view(1, -1, 1, 1)
    
    if t.dim() == 3:
        t = t.unsqueeze(0)  # [1,C,H,W]
    out = t * std + mean
    return out.squeeze(0).clamp(0, 1)

# def preprocess_image(path: str, size: int = 224, equalize: bool = True) -> torch.Tensor:
#     """
#     Load image -> RGB -> optional equalization -> crop -> resize -> normalize (CLIP mean/std).
#     Returns tensor [3,H,W].
#     """
#     bgr = cv2.imread(path, cv2.IMREAD_COLOR)
#     rgb = _ensure_rgb(bgr)

#     if equalize:
#         img_yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
#         img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
#         rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

#     rgb = _center_crop_to_square(rgb)
#     rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    
#     # Convert to float tensor in [0,1]
#     arr = rgb.astype(np.float32) / 255.0
#     chw = torch.from_numpy(arr).permute(2, 0, 1)
    
#     # Apply CLIP normalization
#     chw = TF.normalize(chw, mean=CLIP_MEAN, std=CLIP_STD)
#     return chw


import mediapipe as mp

# Initialize MediaPipe face detector once
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)


def preprocess_image(path: str, size: int = 224, equalize: bool = True) -> torch.Tensor:
    """
    Load image -> detect face using MediaPipe -> crop -> equalize (optional) -> resize -> normalize (CLIP mean/std).
    Returns tensor [3,H,W].
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    # Run MediaPipe face detection
    results = face_detector.process(rgb)
    
    # --- Select best detection if available ---
    if False:#results.detections:
        # pick the largest face (by area)
        boxes = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x0 = int(bbox.xmin * w)
            y0 = int(bbox.ymin * h)
            x1 = int((bbox.xmin + bbox.width) * w)
            y1 = int((bbox.ymin + bbox.height) * h)
            boxes.append((x0, y0, x1, y1))
        # pick largest bounding box
        x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

        # Slightly enlarge to include jawline
        pad_w = int(0.01 * (x1 - x0))
        pad_h = int(0.01 * (y1 - y0))
        x0 = max(0, x0 - pad_w)
        y0 = max(0, y0 - pad_h)
        x1 = min(w, x1 + pad_w)
        y1 = min(h, y1 + pad_h)

        face_rgb = rgb[y0:y1, x0:x1]
    else:
        # fallback to full frame
        face_rgb = rgb

    # Optional Y-channel equalization for lighting normalization
    if equalize:
        img_yuv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        face_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Resize for CLIP input
    face_rgb = cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_AREA)

    # Convert to tensor and normalize
    arr = face_rgb.astype(np.float32) / 255.0
    chw = torch.from_numpy(arr).permute(2, 0, 1)
    chw = TF.normalize(chw, mean=CLIP_MEAN, std=CLIP_STD)
    return chw




def resolve_image_paths(input_path: str) -> List[str]:
    """Collect image file paths from a directory, a single file, or a text list."""
    if os.path.isdir(input_path):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
        files.sort()
        return files
    if os.path.isfile(input_path):
        if input_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            return [input_path]
        # If it's a text file containing one path per line
        if input_path.lower().endswith((".txt", ".lst")):
            with open(input_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return lines
    raise FileNotFoundError(f"Could not resolve images from: {input_path}")

# ==========================
# Batched inference for images
# ==========================

def _tensor_rgb_to_bgr_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: [C,H,W] in [0,1] or uint8, or [1,C,H,W].
    returns BGR uint8 HxWx3 for OpenCV.
    """
    if img_t.dim() == 4:
        img_t = img_t[0]
    if img_t.dtype != torch.uint8:
        img_t = (img_t.clamp(0,1) * 255).to(torch.uint8)
    # CHW -> HWC RGB -> BGR
    return cv2.cvtColor(img_t.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR)

def _detect_yunet_boxes_bgr(bgr: np.ndarray):
    """
    Returns a list of (x, y, w, h, score). Empty list if none.
    Assumes a global/module-level `yunet` already created.
    """
    H, W = bgr.shape[:2]
    yunet.setInputSize((W, H))
    out = yunet.detect(bgr)
    dets = out[1] if isinstance(out, tuple) else out
    if dets is None:
        return []
    dets = np.array(dets, dtype=np.float32)  # N x 15  [x,y,w,h,score, 10 lm coords]
    return [(float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])) for d in dets]

def _shrink_box_anisotropic(xywh, im_w, im_h,
                            shrink_left=0.12, shrink_top=0.35,
                            shrink_right=0.12, shrink_bottom=0.10):
    """
    Shrink (x,y,w,h) more at the top to remove hair. Returns clamped xyxy ints.
    """
    x, y, w, h = xywh
    x0 = x + shrink_left  * w
    y0 = y + shrink_top   * h
    x1 = (x + w) - shrink_right  * w
    y1 = (y + h) - shrink_bottom * h

    x0 = max(0, int(np.floor(x0))); y0 = max(0, int(np.floor(y0)))
    x1 = min(im_w, int(np.ceil(x1))); y1 = min(im_h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:  # fallback to original xyxy
        return int(x), int(y), int(x + w), int(y + h)
    return x0, y0, x1, y1

def _crop_tensor_xyxy(img_t: torch.Tensor, box_xyxy):
    """
    img_t: [C,H,W] or [1,C,H,W]; box: (x0,y0,x1,y1). Returns [C,h,w].
    """
    if img_t.dim() == 4:  # [1,C,H,W] -> [C,H,W]
        img_t = img_t[0]
    C, H, W = img_t.shape
    x0, y0, x1, y1 = map(int, box_xyxy)
    x0 = max(0, min(W-1, x0)); x1 = max(1, min(W, x1))
    y0 = max(0, min(H-1, y0)); y1 = max(1, min(H, y1))
    if x1 <= x0 or y1 <= y0:  # empty -> return original
        return img_t
    return img_t[:, y0:y1, x0:x1]

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def diff_on_white(real_t: torch.Tensor,
                  fake_t: torch.Tensor,
                  thresh: float = 0.02,
                  gamma: float = 0.8,
                  use_grayscale: bool = True) -> torch.Tensor:
    """
    real_t, fake_t: (3,H,W) float in [0,1]
    thresh: pixels with diff magnitude < thresh are set to white
    gamma : contrast boost for the visible differences
    use_grayscale: if True, show magnitude as gray; else keep per-channel diffs
    returns: (3,H,W) float in [0,1]
    """
    # per-pixel absolute difference in [0,1]
    diff = (real_t - fake_t).abs().clamp(0, 1)

    if use_grayscale:
        # a single-channel magnitude => replicate to 3 channels
        mag = diff.max(dim=0).values                        # (H,W)
        if (mag.max() > 0):
            mag = (mag / (mag.max() + 1e-8)).pow(gamma)
        vis = mag.unsqueeze(0).expand(3, -1, -1).clone()    # (3,H,W)
        bg_mask = (mag < thresh)                            # (H,W) bool
        vis[:, bg_mask] = 1.0                               # white background
    else:
        # keep RGB differences, boost contrast a bit
        if (diff.max() > 0):
            diff = (diff / (diff.max() + 1e-8)).pow(gamma)
        bg_mask = (diff < thresh).all(dim=0)                # (H,W) bool
        vis = diff.clone()
        vis[:, bg_mask] = 1.0

    return vis.clamp(0, 1)
def infer_images(
    image_paths: List[str],
    model: MMModerator,
    device: torch.device,
    batch_size: int = 32,
    label: int | None = None,
    equalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run batched inference on images.

    Returns
    -------
    scores : np.ndarray, shape (N,)  — sigmoid(logits) per image
    preds  : np.ndarray, shape (N,)  — thresholded class (0/1) per image
    labels : np.ndarray, shape (N,)  — provided label per image or -1 if not provided
    """
    scores: List[float] = []
    preds: List[int] = []
    labels_out: List[int] = []

    N = len(image_paths)
    i = 0
    while i < N:
        batch_paths = image_paths[i:i+1]
        print(batch_paths)

        batch_tensors = [preprocess_image(p, size=224, equalize=equalize) for p in batch_paths]
        images = torch.stack(batch_tensors, dim=0)  # (B, C, H, W)

        # Build a dummy augmentation copy (or plug your augmentation here)
        images_aug = images.clone()

        # Optional labels (vector of shape [B])
        if label is None:
            label_vec = torch.full((images.size(0),), -1.0)
        else:
            label_vec = torch.full((images.size(0),), float(label))

        images = images.to(device, non_blocking=True)
        images_aug = images_aug.to(device, non_blocking=True)
        label_vec = label_vec.to(device)

        with torch.no_grad():
            # Forward — using the image fields of your model
            # logits_cls, image_recon,generated_captions,captions_readable, overlay
            # The model should accept images/images_aug (and ignore others)
            
            logits,  image_recon, generated_captions, captions_readable, overlay  = model(
                mfcc=None,
                mfcc_aug=None,
                audio=None,
                video=None,
                video_aug=None,
                text=None,
                landmarks=None,
                flow=None,
                images=images,
                images_aug=images_aug,
                labels=label_vec,
                multi_label=label_vec.long() if label is not None else None,
            )

            # logits, losses,labels_expanded, image_recon, generated_captions,captions, captions_readable, overlay, overlay_ori  = model(
            #     mfcc=None,
            #     mfcc_aug=None,
            #     audio=None,
            #     video=None,
            #     video_aug=None,
            #     text=None,
            #     landmarks=None,
            #     flow=None,
            #     images=images,
            #     images_aug=images_aug,
            #     labels=label_vec,
            #     multi_label=label_vec.long() if label is not None else None,
            # )
            fake_path = image_paths[i]
            file = os.path.basename(fake_path)
            base_path = os.path.dirname(fake_path)
            pair = os.path.basename(base_path)
            
            method = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(base_path))))

            image = unnormalize(images[0])
            name = os.path.basename(Path(batch_paths[0]).parent)
            to_pil(image).save(f"./temp/overlays/{name}_{i}_img.jpg")
            
            to_pil(overlay[0].clamp(0, 1)).save(f"./temp/overlays/{name}_{i}_overlay.jpg")

            probs = sigmoid(logits).detach().flatten().cpu()  # (B,)
            batch_scores = probs.numpy().tolist()
            batch_preds = (probs > 0.5).long().cpu().numpy().tolist()

        scores.extend(batch_scores)
        preds.extend(batch_preds)
        labels_out.extend([int(label) if label is not None else -1] * len(batch_paths))
        i += 1

    return np.array(scores, dtype=np.float32), np.array(preds, dtype=np.int64), np.array(labels_out, dtype=np.int64), generated_captions, captions_readable

# ==========================
# Optional: directory with class subfolders
# ==========================

def collect_labeled_images(root: str) -> Tuple[List[str], List[int]]:
    """
    Expect a directory like:
      root/
        real/   *.jpg
        fake/   *.jpg
    Returns (paths, labels) with label 0 for real, 1 for fake.
    """
    paths: List[str] = []
    labels: List[int] = []
    for cls, y in [("real", 0), ("fake", 1)]:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        files = resolve_image_paths(d)
        paths.extend(files)
        labels.extend([y] * len(files))
    return paths, labels

# ==========================
# CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Batched image inference with MMModerator")
    parser.add_argument("--images_path", type=str, required=True,
                        help="Path to a single image, a directory of images, or a .txt list of paths.\n"
                             "Alternatively, pass a directory that contains 'real' and 'fake' subfolders for labeled eval.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--equalize", action="store_true", help="Apply Y-channel histogram equalization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Case A: Labeled folder with real/ and fake/
    if (os.path.isdir(args.images_path)
            and os.path.isdir(os.path.join(args.images_path, "real"))
            and os.path.isdir(os.path.join(args.images_path, "fake"))):
        print("[*] Detected labeled directory structure (real/ and fake/). Running evaluation…")
        all_paths, all_labels = collect_labeled_images(args.images_path)
        if not all_paths:
            raise RuntimeError("No images found under the labeled directory.")

        # We'll run in a single pass mixing labels; to supply labels per-batch, we chunk by class ourselves.
        # Simpler approach: run twice (real then fake) and concatenate.
        real_paths = [p for p, y in zip(all_paths, all_labels) if y == 0]
        fake_paths = [p for p, y in zip(all_paths, all_labels) if y == 1]

        s_real, p_real, y_real, captions_readable_real = infer_images(real_paths, model, device, args.batch_size, label=0, equalize=args.equalize)
        s_fake, p_fake, y_fake , captions_readable_fake = infer_images(fake_paths, model, device, args.batch_size, label=1, equalize=args.equalize)

        scores = np.concatenate([s_real, s_fake])
        preds  = np.concatenate([p_real, p_fake])
        labels = np.concatenate([y_real, y_fake])

        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds)
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float('nan')
        ap  = average_precision_score(labels, scores)

        print("\n===== IMAGE-LEVEL METRICS =====")
        print(f"Accuracy          : {acc:.4f}")
        print(f"F1-score          : {f1:.4f}")
        print(f"ROC-AUC           : {auc:.4f}")
        print(f"Average Precision : {ap:.4f}")

        # Save a TSV of results
        out_tsv = os.path.join(args.images_path, "image_scores.tsv")
        with open(out_tsv, "w") as f:
            f.write("path\tscore\tpred\tlabel\n")
            for p, s, pr, y in zip(list(real_paths) + list(fake_paths), scores, preds, labels):
                f.write(f"{p}\t{s:.6f}\t{int(pr)}\t{int(y)}\n")
        print(f"Saved results to {out_tsv}")

    # Case B: Unlabeled (single image / dir / list)
    else:
        img_paths = resolve_image_paths(args.images_path)

        scores, preds, labels, generated_captions, captions_readable = infer_images(img_paths, model, device, args.batch_size, label=None, equalize=args.equalize)
        for p, s, pr in zip(img_paths, scores, preds):
            print(f"{p}\tP(fake)={s:.4f}\tpred={int(pr)}")
            


if __name__ == "__main__":
    main()
