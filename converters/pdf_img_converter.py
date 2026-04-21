import fitz
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PDFImageConverter:
    """
    Chuyển đổi từng trang PDF thành ảnh, hỗ trợ lazy loading để tiết kiệm RAM.
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.zoom = dpi / 72  # 72 là DPI mặc định của PDF

    def iter_pages(self, pdf_path: str):
        """
        Generator lazy load: yield từng (page_index, fitz.Page) một.
        Đảm bảo doc.close() luôn được gọi dù caller break hay exception.

        Usage:
            for page_idx, page in converter.iter_pages(pdf_path):
                # xử lý page
        """
        doc = fitz.open(pdf_path)
        try:
            for index in range(len(doc)):
                yield index, doc.load_page(index)
        finally:
            doc.close()

    def page_to_cv2(self, page: fitz.Page) -> np.ndarray:
        """
        Render fitz.Page thành ảnh OpenCV (BGR).
        """
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def save_page_as_image(self, page: fitz.Page, output_path: str):
        """
        Lưu fitz.Page thành file PNG.
        """
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=matrix)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pix.save(output_path)

    def get_page_count(self, pdf_path: str) -> int:
        """
        Trả về số trang của PDF mà không load toàn bộ document.
        """
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    def extract_embedded_images(self, pdf_path: str, output_dir: str, doc_id: int) -> dict:
        """
        Extract ảnh nhúng gốc từ PDF (không qua bitmap).
        Trả về dict: {(page_idx, xref): "images/doc_N_image_M.png"}

        Dùng cho các page loại normal image (không phải rasterized/scan).
        """
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        image_map = {}
        order = 1

        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        ext = base_image["ext"]

                        img_filename = f"doc_{doc_id}_image_{order}.png"
                        img_path = os.path.join(images_dir, img_filename)

                        if ext.lower() == "png":
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                        else:
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if cv_img is not None:
                                cv2.imwrite(img_path, cv_img)
                            else:
                                continue

                        image_map[(page_idx, xref)] = f"images/{img_filename}"
                        order += 1

                    except Exception as e:
                        print(f"[WARN] Không extract được ảnh xref={xref}: {e}")
        finally:
            doc.close()

        return image_map
    
    def show_image(self, img: np.ndarray, title: str = "Debug Image"):
        """
        Hiển thị ảnh bằng matplotlib — có thể zoom, pan thoải mái.
        """
        if img is None:
            print(f"[DEBUG] {title}: ảnh bị None")
            return

        print(f"[DEBUG] {title}: shape={img.shape}, dtype={img.dtype}")

        # OpenCV dùng BGR, matplotlib dùng RGB
        if len(img.shape) == 3:
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_show = img  # Grayscale

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img_show, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=12)
        ax.axis('on')  # Bật axis để thấy tọa độ khi zoom
        plt.tight_layout()
        plt.show()


    def show_page(self, page: fitz.Page, title: str = "Page Preview"):
        """
        Render fitz.Page và hiển thị bằng matplotlib.
        """
        img = self.page_to_cv2(page)
        self.show_image(img, title=title)


    def show_regions(self, img: np.ndarray, regions: list, title: str = "Regions"):
        """
        Hiển thị ảnh kèm bounding box của từng region — debug OCR result.
        Mỗi label có màu khác nhau.
        """
        if img is None:
            return

        LABEL_COLORS = {
            "table":           "red",
            "figure":          "blue",
            "image":           "cyan",
            "paragraph":       "green",
            "paragraph_title": "orange",
            "doc_title":       "purple",
            "figure_title":    "pink",
            "header":          "gray",
            "footer":          "gray",
            "text":            "lime",
            "vision_footnote": "brown",
        }

        if len(img.shape) == 3:
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_show = img

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img_show, cmap='gray' if len(img.shape) == 2 else None)

        for region in regions:
            label = region.get('label', 'unknown')
            bbox = region.get('bbox', None)
            content = region.get('content', '')[:30]  # Preview 30 ký tự

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            # Scale bbox về kích thước ảnh đã render
            color = LABEL_COLORS.get(label, "yellow")

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5,
                f"[{label}] {content}",
                fontsize=7,
                color=color,
                backgroundcolor='black'
            )

        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        plt.show()

