import fitz
import os
import cv2
import re
import numpy as np
import pandas as pd
from paddleocr import PPStructureV3
from bs4 import BeautifulSoup
from tabulate import tabulate
from utils.denoise import preprocess_img #,denoise_img

class Img2MdConverter:
    def __init__(self, engine: PPStructureV3):
        self.engine = engine

    # - Case past only drug prove.
    # -Ремень салон устройство слишком холодно похороны июнь. <- issue
    # о
    # Young however many.
    # def is_bullet_text(self, content: str) -> bool:
    #     stripped = content.strip()
    #     return bool(re.match(r'^[•◦▪\-\*–·]\s', stripped))
    
    def is_bullet_text(self, content: str) -> bool:
        """
        Kiểm tra text có phải list item không dựa vào ký tự đầu dòng.
        """
        stripped = content.strip()
        # Match cả có space lẫn không có space sau bullet
        return bool(re.match(r'^[•◦▪\-\*–·о]\s*\S', stripped))


    def clean_bullet(self, content: str) -> str:
        """Xóa ký tự bullet đầu dòng và khoảng trắng thừa"""
        # Xóa bullet marker + khoảng trắng phía sau
        content = re.sub(r'^[•◦▪\-\*–·о]\s*', '', content.strip())
        # Xóa các ký tự rác đầu chuỗi (như "HЕ ДЛЯ Р-")
        # content = re.sub(r'^[А-ЯA-Z\s]+-\s*', '', content) if re.match(r'^[А-ЯA-Z\s]+-\s+[а-яa-z]', content) else content
        return content.strip()
    
    def cluster_x_positions(self, x_positions: list) -> dict:
        sorted_x = sorted(set(x_positions))
        
        if len(sorted_x) == 1:
            return {sorted_x[0]: 0}

        # Tính threshold động = 50% của khoảng cách nhỏ nhất giữa các x
        gaps = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
        min_gap = min(gaps)
        max_gap = max(gaps)
        
        # Nếu gap quá đều → dùng median, không thì dùng min_gap * 1.5
        threshold = max(30, min_gap * 1.5)
        
        # print(f"[DEBUG] x_positions={sorted_x}, threshold={threshold:.0f}")

        clusters = []
        current = [sorted_x[0]]

        for x in sorted_x[1:]:
            if x - current[-1] <= threshold:
                current.append(x)
            else:
                clusters.append(current)
                current = [x]
        clusters.append(current)

        x_to_level = {}
        for level, cluster in enumerate(clusters):
            for x in cluster:
                x_to_level[x] = level

        return x_to_level
    
    def detect_list_levels(self, regions: list) -> list:
        RESET_LABELS = {"doc_title", "paragraph_title", "section_title",
                        "table", "image", "figure_title"}

        sorted_regions = sorted(regions, key=lambda r: r.get('bbox', [0,0,0,0])[1])

        groups = []
        current_group = []

        for r in sorted_regions:
            if r.get('label') in RESET_LABELS:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([r])
            else:
                current_group.append(r)

        if current_group:
            groups.append(current_group)

        for group in groups:
            text_items = [r for r in group if r.get('label') == 'text']
            if not text_items:
                for r in group:
                    r['list_level'] = 0
                    r['is_list_group'] = False
                continue

            x_positions = [r['bbox'][0] for r in text_items]
            x_to_level = self.cluster_x_positions(x_positions)

            # Group có nhiều hơn 1 x level → đây là list có phân cấp
            is_list = len(set(x_to_level.values())) > 1

            # Hoặc có bất kỳ item nào có bullet marker → cả group là list
            has_bullet = any(self.is_bullet_text(r.get('content', '')) for r in text_items)
            is_list = is_list or has_bullet

            for r in group:
                if r.get('label') == 'text':
                    x = r['bbox'][0]
                    closest_x = min(x_to_level.keys(), key=lambda px: abs(px - x))
                    r['list_level'] = x_to_level[closest_x]
                    r['is_list_group'] = is_list
                else:
                    r['list_level'] = 0
                    r['is_list_group'] = False

        return sorted_regions

    def image_to_html(self, page: fitz.Page) -> list[str]:
        denoised_img = preprocess_img(page)
        result = self.engine.predict(denoised_img)

        parsed_regions = []


        for page_result in result:
            regions = page_result.get("parsing_res_list", [])
            for region in regions:
                # Convert LayoutBlock sang dict
                parsed_regions.append({
                    "index":        region.index,
                    "label":        region.label,
                    "bbox":         list(region.bbox),
                    "content":      region.content,
                    "region_label": getattr(region, "region_label", ""),
                })

        return parsed_regions  # List of dict: label, content, bbox, index

        
    def html_to_markdown_table(self, html_content: str):
        """Chuyển đổi table HTML (kể cả ô gộp) sang Markdown."""
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        if not table:
            return ""

        rows = []
        max_cols = 0
        
        # Duyệt qua các hàng (tr)
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                # Lấy nội dung text và làm sạch
                text = td.get_text(separator=" ", strip=True)
                cells.append(text)
            rows.append(cells)
            max_cols = max(max_cols, len(cells))

        # Đảm bảo các hàng có số cột bằng nhau (tránh lỗi tabulate)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        return tabulate(rows, headers="firstrow", tablefmt="pipe")

    def process_document_to_markdown(self, data_list: list) -> str:
        SKIP_LABELS = {"header", "footer", "number", "aside_text",
                       "header_image", "footer_image"}
        GARBAGE_CHARS = {'о', 'О', 'o', 'O', '·', '•', '-', '—'}

        # Detect list levels
        data_list = self.detect_list_levels(data_list)

        # Sort: trên xuống dưới, trái sang phải
        def sort_key(item):
            bbox = item.get('bbox', [0, 0, 0, 0])
            return (bbox[1] // 100, bbox[0])

        sorted_items = sorted(data_list, key=sort_key)
        markdown_output = []

        for item in sorted_items:
            label   = item.get('label', '')
            content = item.get('content', '').strip()
            level   = item.get('list_level', 0)
            is_list = item.get('is_list_group', False)

            if not content or label in SKIP_LABELS:
                continue
            if content in GARBAGE_CHARS:
                continue
            if len(content) <= 1 and not content.isalnum():
                continue

            # --- Heading ---
            if label == 'doc_title':
                markdown_output.append(f"# {content}\n")

            elif label in ('paragraph_title', 'section_title'):
                # Numbered sub-heading → ####
                if re.match(r'^\d+[\.\)]', content):
                    markdown_output.append(f"#### {content}\n")
                else:
                    markdown_output.append(f"### {content}\n")

            elif label == 'figure_title':
                # Plain text theo etalon, bỏ ký tự rác trước Рис.
                content = re.sub(r'^[^РрRr]*?(Рис\.|Fig\.)', r'\1', content)
                markdown_output.append(f"### {content}\n")

            # --- Bảng ---
            elif label == 'table':
                if "<table>" in content:
                    md_table = self.html_to_markdown_table(content)
                    if md_table:
                        markdown_output.append(md_table + "\n")
                else:
                    markdown_output.append(content + "\n")

            # --- Ảnh ---
            elif label == 'image':
                markdown_output.append(
                    f"__FIGURE_PLACEHOLDER_{item.get('index', 0)}__\n"
                )

            # --- Text ---
            elif label == 'text':
                # Bỏ prefix rác kiểu "НЕ ДЛЯ Р-"
                content = re.sub(r'^[А-ЯA-Z\s]{2,10}-\s+', '', content)
                if not content.strip():
                    continue

                indent = "  " * level
                if self.is_bullet_text(content):
                    text = self.clean_bullet(content)
                    markdown_output.append(f"{indent}- {text}\n")
                elif is_list:
                    markdown_output.append(f"{indent}- {content}\n")
                else:
                    markdown_output.append(f"{content}\n")

            # --- vision_footnote → bỏ theo etalon ---
            elif label == 'vision_footnote':
                continue

            else:
                markdown_output.append(f"{content}\n")

        return "\n".join(markdown_output)

