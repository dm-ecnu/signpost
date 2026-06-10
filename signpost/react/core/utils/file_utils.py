#
#  Pure file utility functions (no business logic)
#
#  From api/utils/file_utils.py: filename_type, thumbnail*, traversal_files, PDF repair
#
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from io import BytesIO

import pdfplumber
from PIL import Image

from core.config import IMG_BASE64_PREFIX, LOCK_KEY_pdfplumber
from core.db import FileType


def filename_type(filename):
    filename = filename.lower()
    if re.match(r".*\.pdf$", filename):
        return FileType.PDF.value

    if re.match(r".*\.(eml|doc|docx|ppt|pptx|yml|xml|htm|json|csv|txt|ini|xls|xlsx|wps|rtf|hlp|pages|numbers|key|md|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|html|sql)$", filename):
        return FileType.DOC.value

    if re.match(r".*\.(wav|flac|ape|alac|wavpack|wv|mp3|aac|ogg|vorbis|opus)$", filename):
        return FileType.AURAL.value

    if re.match(r".*\.(jpg|jpeg|png|tif|gif|pcx|tga|exif|fpx|svg|psd|cdr|pcd|dxf|ufo|eps|ai|raw|WMF|webp|avif|apng|icon|ico|mpg|mpeg|avi|rm|rmvb|mov|wmv|asf|dat|asx|wvx|mpe|mpa|mp4)$", filename):
        return FileType.VISUAL.value

    return FileType.OTHER.value


def thumbnail_img(filename, blob):
    """MySQL LongText max length is 65535"""
    filename = filename.lower()
    if re.match(r".*\.pdf$", filename):
        with sys.modules[LOCK_KEY_pdfplumber]:
            pdf = pdfplumber.open(BytesIO(blob))
            buffered = BytesIO()
            resolution = 32
            img = None
            for _ in range(10):
                pdf.pages[0].to_image(resolution=resolution).annotated.save(buffered, format="png")
                img = buffered.getvalue()
                if len(img) >= 64000 and resolution >= 2:
                    resolution = resolution / 2
                    buffered = BytesIO()
                else:
                    break
        pdf.close()
        return img

    elif re.match(r".*\.(jpg|jpeg|png|tif|gif|icon|ico|webp)$", filename):
        image = Image.open(BytesIO(blob))
        image.thumbnail((30, 30))
        buffered = BytesIO()
        image.save(buffered, format="png")
        return buffered.getvalue()

    elif re.match(r".*\.(ppt|pptx)$", filename):
        import aspose.pydrawing as drawing
        import aspose.slides as slides

        try:
            with slides.Presentation(BytesIO(blob)) as presentation:
                buffered = BytesIO()
                scale = 0.03
                img = None
                for _ in range(10):
                    presentation.slides[0].get_thumbnail(scale, scale).save(buffered, drawing.imaging.ImageFormat.png)
                    img = buffered.getvalue()
                    if len(img) >= 64000:
                        scale = scale / 2.0
                        buffered = BytesIO()
                    else:
                        break
                return img
        except Exception:
            pass
    return None


def thumbnail(filename, blob):
    img = thumbnail_img(filename, blob)
    if img is not None:
        return IMG_BASE64_PREFIX + base64.b64encode(img).decode("utf-8")
    else:
        return ""


def traversal_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def repair_pdf_with_ghostscript(input_bytes):
    if shutil.which("gs") is None:
        return input_bytes

    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_in, tempfile.NamedTemporaryFile(suffix=".pdf") as temp_out:
        temp_in.write(input_bytes)
        temp_in.flush()

        cmd = ["gs", "-o", temp_out.name, "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/prepress", temp_in.name]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return input_bytes
        except Exception:
            return input_bytes

        temp_out.seek(0)
        repaired_bytes = temp_out.read()

    return repaired_bytes


def read_potential_broken_pdf(blob):
    def try_open(blob):
        try:
            with pdfplumber.open(BytesIO(blob)) as pdf:
                if pdf.pages:
                    return True
        except Exception:
            return False
        return False

    if try_open(blob):
        return blob

    repaired = repair_pdf_with_ghostscript(blob)
    if try_open(repaired):
        return repaired

    return blob
