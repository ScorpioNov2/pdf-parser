import zipfile

with zipfile.ZipFile("D:\pdf-extractor\src-v1\data\output\submission.zip", 'r') as zf:
    print(zf.namelist()[:10])  # Show first 10 file
    bad = zf.testzip()
    print("Bad file:", bad)