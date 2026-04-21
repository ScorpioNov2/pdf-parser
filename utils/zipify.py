import zipfile
from pathlib import Path

output_path = Path("./data/output")
zip_name = "submission.zip"

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED, 
                     allowZip64=True) as zf:
    for md_file in sorted(output_path.glob("*.md")):
        zf.write(md_file, md_file.name)

    images_dir = output_path / "images"
    if images_dir.exists():
        for img_file in sorted(images_dir.glob("*.png")):
            zf.write(img_file, f"images/{img_file.name}")

print("Done")