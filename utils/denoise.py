# Tách ảnh chất lượng cao
import fitz
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fitz import Page

# ─────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────
def estimate_noise(img: np.ndarray) -> float:
    """Salt-and-pepper noise ratio (giữ nguyên logic cũ)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median)
    return float(np.mean(diff > 20))


def preprocess_img(img: np.ndarray) -> np.ndarray:
    ir = estimate_noise(img)
    # print(f"[DEBUG] isolated_ratio={ir:.4f}")

    if ir > 0.04: # Nhiễu cực nặng như ảnh mẫu
        img = cv2.medianBlur(img, 5)
        # plt.imshow(img)
        # plt.show()
        return apply_clahe(img, 3)

        # return img
        
    elif ir > 0.015:
        # print("[DEBUG] → Scan nặng, median blur 3")
        img = cv2.medianBlur(img, 5) # 3
        img = apply_clahe(img, 3)
        # plt.imshow(img)
        # plt.show()
        return img
        # return img

    elif ir > 0.005:
        # print("[DEBUG] → Nhiễu nhẹ, median blur 3")
        img = cv2.medianBlur(img, 3) # 3
        img = apply_clahe(img, 2)
        # plt.imshow(img)
        # plt.show()
        return img
        # return apply_clahe(img, clip=3.0)
    else:
        return img

def apply_clahe(img, clip=2.0):
    # Hàm bổ trợ để tăng tương phản mà không làm hỏng ảnh
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit càng cao, chữ càng đen và rõ
    # tileGridSize giữ (8,8) là chuẩn nhất cho văn bản
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    
    l = clahe.apply(l)
    img_res = cv2.merge((l, a, b))
    return cv2.cvtColor(img_res, cv2.COLOR_LAB2BGR)

