import re

def replace_figure_placeholders(md: str, image_map: dict, page_idx: int) -> str:
    """
    Thay __FIGURE_PLACEHOLDER_N__ bằng link ảnh thật hoặc xóa nếu không có ảnh nhúng.

    Args:
        md: markdown string chứa placeholder
        image_map: dict từ extract_embedded_images {(page_idx, xref): "images/doc_N_image_M.png"}
        page_idx: index của page hiện tại
    """
    # Lấy danh sách ảnh thuộc page này, theo thứ tự xref
    page_images = [
        img_path
        for (p_idx, xref), img_path in sorted(image_map.items(), key=lambda x: x[0][1])
        if p_idx == page_idx
    ]

    # Iterator để lấy từng ảnh theo thứ tự
    image_iter = iter(page_images)

    def replacer(match: re.Match) -> str:
        img_path = next(image_iter, None)
        if img_path:
            # Có ảnh nhúng → insert link
            return f"![image]({img_path})"
        else:
            # Không có ảnh nhúng → rasterized/scan → bỏ placeholder
            return ""

    result = re.sub(r"__FIGURE_PLACEHOLDER_\d+__", replacer, md)

    # Dọn dẹp dòng trống thừa
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result