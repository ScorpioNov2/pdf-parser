from converters.img_md_converter import Img2MdConverter
from converters.pdf_img_converter import PDFImageConverter
from paddleocr import PPStructureV3
import fitz
import os
from pathlib import Path
from utils.denoise import preprocess_img
from utils.replace_figure_placeholders import replace_figure_placeholders

# properties = paddle.device.cuda.get_device_properties(0)
# print(f"GPU: {properties.name}") # GPU: NVIDIA GeForce RTX 4060 Laptop GPU

class Config:
    PDF_DIR_PATH = "data/input"
    OUPUT_DIR_PATH = "data/output"
    MD_DIR_PATH = "data/output"
    IMAGES_DIR_PATH = "data/output/images_raw"

# Debug crop vùng bảng
def show_crop(converter, img, bbox, title="Crop"):
    x1, y1, x2, y2 = bbox
    # Scale bbox về kích thước ảnh đã render (dpi=300 → zoom=300/72)
    crop = img[y1:y2, x1:x2]
    converter.show_image(crop, title=title)

if __name__ == "__main__":
    # engine = PPStructureV3(
    #     lang="ru",
    #     # ocr_version='PP-OCRv3',      # Comment to auto handle, v4 not support ru but v3 multi languages
    #     use_table_recognition=True,
    #     use_region_detection=True,
    #     use_formula_recognition=True,
    #     enable_mkldnn=False,
    #     device="gpu:0"
    #     # use_gpu=True, 
    #     # gpu_id=1 # ValueError: Unknown argument: gpu_id
    # )

    # engine = PPStructureV3(
    #     lang="ru",
    #     use_table_recognition=True,  # Tắt tạm để test
    #     use_region_detection=True,
    #     use_formula_recognition=True,  # Tắt tạm để test
    #     enable_mkldnn=False,
    #     device="gpu:0"
    # )

    # engine = PPStructureV3(
    #     text_recognition_model_name="cyrillic_PP-OCRv3_mobile_rec",
    #     use_table_recognition=True,
    #     use_region_detection=True,
    #     use_formula_recognition=True,
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     device="gpu:0"
    # )

    engine = PPStructureV3(
        # text_recognition_model_name="ru_PP-OCRv5_mobile_rec", # not supported in this version
        # text_detection_model_name="PP-OCRv5_server_det",    # Server det
        # text_recognition_model_name="PP-OCRv5_server_rec",  # Server rec (default)
        lang="ru",
        use_table_recognition=True,
        use_region_detection=True,
        use_formula_recognition=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        device="gpu:0",

        # text_det_limit_side_len=4000,
        # text_det_limit_type="max",
        # text_det_thresh=0.2, 
        # text_det_box_thresh=0.3,
        # text_det_unclip_ratio=2.5,
    )
    config = Config()
    pdf_to_img_converter = PDFImageConverter(dpi=300)
    img_to_md_converter = Img2MdConverter(engine=engine)

    # pdf_path = config.PDF_DIR_PATH + "/document_005.pdf"
    # pdf_dir = Path(config.PDF_DIR_PATH)
    pdf_dir = Path(config.PDF_DIR_PATH)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))  # Sort để đúng thứ tự
    # pdf_files=pdf_files[19:]
    print(f"Number of files in {config.PDF_DIR_PATH}: {len(pdf_files)}")

    # from utils.test_denoise import medianFiltering
    # precessed_pic = medianFiltering()
    # html = img_to_md_converter.image_to_html(precessed_pic)
    # print(html)

    for i, pdf_path in enumerate(pdf_files, start=1):
        # print(f"[{i}/{len(pdf_files)}] In processing: {pdf_path.name}")

        # get doc id
        doc_id = int(pdf_path.stem.split("_")[-1])

        try:
            image_map = pdf_to_img_converter.extract_embedded_images(str(pdf_path), config.OUPUT_DIR_PATH, doc_id=doc_id)

            for page_idx, page in pdf_to_img_converter.iter_pages(str(pdf_path)):
                # Debug ảnh gốc
                # pdf_to_img_converter.show_page(page, title=f"Page {page_idx + 1} - Raw")

                # Preprocess
                img_raw = pdf_to_img_converter.page_to_cv2(page)
                img_processed  = preprocess_img(img_raw)  # nhận np.ndarray thay vì fitz.Page

                # Dùng bbox từ result
                # show_crop(pdf_to_img_converter, img_raw, [362, 370, 2276, 1156], "Page 4 Table - Raw")
                # show_crop(pdf_to_img_converter, img_processed, [362, 370, 2276, 1156], "Page 4 Table - Processed")

                # Debug sau preprocess
                # pdf_to_img_converter.show_image(img_processed, title=f"Page {page_idx + 1} - Processed")

                # OCR
                regions = img_to_md_converter.image_to_html(img_processed)
                # print("[debug regions]")
                # print(regions)
                # Xem regions OCR detect được — cái này hữu ích nhất
                # pdf_to_img_converter.show_regions(img_processed, regions, title=f"Page {page_idx + 1} Regions")
                # print(regions)

                md = img_to_md_converter.process_document_to_markdown(regions)

                # Replace figure placeholder với link ảnh thật
                md = replace_figure_placeholders(md, image_map, page_idx)

                # Ghi file
                with open(f"data/output/{pdf_path.stem}.md", "a", encoding="utf-8") as f:
                    f.write(md)
                
            # print(f"[+] Completed page {page_idx + 1} in file {pdf_path.stem}.pdf")

            # print(f"Saved to data/output/{pdf_path.stem}.md")
            # break
        except Exception as e:
            print(f"  [ERROR] {pdf_path.name}: {e}")
            continue
            # break
        if(doc_id % 20 == 0):
            print(f"[+] Completed page {page_idx + 1} in file {pdf_path.stem}.pdf")

    print(f"Saved to data/output/{pdf_dir}.md")


    # =========================================== V1 ============================================
    # pdf_to_img_converter = PDFImageConverter()
    # doc = fitz.open(pdf_path)
    # final_content = []
    # for page_idx in range(len(doc)):
    # for page_idx in range(1):
    #     # page = doc[page_idx]
    #     bitmap_page= pdf_to_img_converter.convert_to_images(pdf_path=pdf_path, output_folder=config.MD_DIR_PATH, dpi=300)
    #     # page_md = img_to_md_converter.html_to_markdown_table(img_to_md_converter.image_to_html(page=bitmap_page))
    #     # page_md = img_to_md_converter.process_document_to_markdown(img_to_md_converter.image_to_html(page=bitmap_page))
    #     page_md = img_to_md_converter.process_document_to_markdown(img_to_md_converter.image_to_html(page=(fitz.open("D:\\pdf-extractor\\src-v1\\test.png").load_page(0))))

    #     final_content.append(page_md)
    #     print(f"[+] Hoàn thành trang {page_idx + 1}")

    # # Ghi file kết quả
    # with open(f"data/output/file_{1}.md", "a", encoding="utf-8") as f:
    #     f.write("\n\n".join(final_content))
    #     # with open(f"data/output/file_{1}.md", "a", encoding="utf-8") as f:
    #     #     f.write("\n\n".join(final_content))
        
    # doc.close()
    # ===========================================================================================

