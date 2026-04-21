# Tách ảnh chất lượng cao
import fitz
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fitz import Page
from utils.denoise_methods import DenoiseMethods

dn_methods = DenoiseMethods()

# ─────────────────────────────────────────────
# 2. Metrics đánh giá ảnh
# ─────────────────────────────────────────────
def estimate_noise(img: np.ndarray) -> float:
    """Salt-and-pepper noise ratio (giữ nguyên logic cũ)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median)
    return float(np.mean(diff > 20))


def estimate_background_texture(gray: np.ndarray) -> float:
    """
    Đo mức độ không đồng đều của nền (texture/gradient).
    Dùng để quyết định có cần background normalization không.
    
    Ý tưởng: ước lượng nền bằng morphological closing kernel lớn,
    rồi đo độ lệch chuẩn của nền ước lượng đó.
    Nền sạch → std thấp; nền texture/gradient → std cao.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    background_est = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return float(np.std(background_est))


# ─────────────────────────────────────────────
# 3. Các bước xử lý ảnh cơ bản
# ─────────────────────────────────────────────
def normalize_background(gray: np.ndarray) -> np.ndarray:
    """
    Loại bỏ texture nền không đồng đều (hiệu quả nhất với ảnh document).
    
    Thuật toán:
      1. Ước lượng nền bằng morphological closing kernel rất lớn
         (kernel lớn → chỉ giữ lại biến thiên chậm = nền, không lấy text)
      2. Chia ảnh gốc cho nền ước lượng → normalize illumination
      3. Kết quả: text đen trên nền trắng đều, bất kể texture ban đầu
    
    Đây là bước quan trọng nhất với các PDF có nền hạt/texture.
    """
    # Kernel size = ~1/6 chiều rộng ảnh, tối thiểu 51 (phải lẻ)
    h, w = gray.shape
    ksize = max(51, (w // 6) | 1)  # | 1 để đảm bảo lẻ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    
    # Ước lượng nền: closing lấy phần sáng nhất trong vùng lớn
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Normalize: pixel_result = (pixel_goc / background) * 255
    # cv2.divide tránh overflow và xử lý division-by-zero tự động
    normalized = cv2.divide(gray.astype(np.float32),
                            background.astype(np.float32),
                            scale=255.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def denoise(gray: np.ndarray, noise_level: float) -> np.ndarray:
    """
    fastNlMeansDenoising tốt hơn medianBlur vì:
    - Bảo toàn cạnh text tốt hơn
    - Loại bỏ nhiễu Gaussian + salt-and-pepper đồng thời
    - h=filter strength: tăng theo mức độ nhiễu
    """
    if noise_level > 0.04:
        h_strength = 18
    elif noise_level > 0.015:
        h_strength = 12
    elif noise_level > 0.005:
        h_strength = 7
    else:
        return gray  # Ảnh đã sạch, không cần denoise

    return cv2.fastNlMeansDenoising(
        gray,
        h=h_strength,
        templateWindowSize=7,   # Vùng so sánh patch: 7x7
        searchWindowSize=21,    # Vùng tìm kiếm: 21x21
    )


def binarize_adaptive(gray: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Adaptive thresholding tốt hơn global Otsu khi:
    - Còn chút biến thiên sáng cục bộ sau normalize
    - Text có kích thước khác nhau trên cùng trang
    
    blockSize: vùng cục bộ để tính ngưỡng (phải lẻ, ~31-51px ở 300dpi)
    C: hằng số trừ khỏi ngưỡng (tăng C → loại bỏ nhiều nền hơn)
    """
    if noise_level > 0.015:
        block_size, C = 35, 12
    else:
        block_size, C = 25, 8

    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )


def sharpen_text(gray: np.ndarray) -> np.ndarray:
    """
    Unsharp masking: làm sắc nét cạnh text cho ảnh sạch.
    Chỉ áp dụng khi ảnh ít nhiễu (tránh làm sắc cả nhiễu).
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    # unsharp = original + alpha * (original - blurred)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Chỉnh nghiêng nhẹ (< 5 độ) — phổ biến với scan PDF.
    Dùng Hough line detection để tìm góc nghiêng của text.
    """
    # Tìm cạnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough lines để tìm đường ngang
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100,
                             minLineLength=gray.shape[1] // 4,
                             maxLineGap=20)
    if lines is None:
        return gray

    # Tính góc trung bình của các đường gần nằm ngang
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 5:  # Chỉ lấy đường gần nằm ngang
                angles.append(angle)

    if not angles:
        return gray

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:  # Quá nhỏ, bỏ qua
        return gray

    # Xoay ảnh
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ─────────────────────────────────────────────
# 4. Pipeline chính
# ─────────────────────────────────────────────
def preprocess_img(img: np.ndarray) -> np.ndarray:
    """
    Pipeline xử lý ảnh document cho PaddleOCR.
    
    Luồng xử lý:
      BGR → Gray → [Deskew] → [BG Normalize] → [Denoise] → [Binarize/Sharpen] → BGR
    
    Trả về BGR vì PPStructureV3 expect ảnh màu.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Đo metrics ──────────────────────────────────────────
    noise_level = estimate_noise(img)
    bg_texture = estimate_background_texture(gray)

    # ── Bước 1: Deskew (luôn chạy, rẻ và an toàn) ──────────
    gray = deskew(gray)

    # ── Bước 2: Background normalization ────────────────────
    # bg_texture > 8 = nền không đồng đều đáng kể
    # (các PDF sample của bạn đều cần bước này)
    if bg_texture > 8.0:
        gray = normalize_background(gray)

    # ── Bước 3: Denoising ────────────────────────────────────
    gray = denoise(gray, noise_level)

    # ── Bước 4: Binarize hoặc Sharpen ────────────────────────
    if noise_level > 0.005 or bg_texture > 8.0:
        # Ảnh có nhiễu hoặc texture nền → binarize
        gray = binarize_adaptive(gray, noise_level)
        
        # Morphological cleanup: xóa chấm nhiễu nhỏ còn sót
        # erosion nhẹ → loại pixel đơn lẻ
        # dilation → phục hồi độ dày nét chữ
        kernel_clean = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_clean)
    else:
        # Ảnh sạch → chỉ sharpen để tăng độ rõ nét
        gray = sharpen_text(gray)

    # Trả về BGR (PaddleOCR yêu cầu)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

