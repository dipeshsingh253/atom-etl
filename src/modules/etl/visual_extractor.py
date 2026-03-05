"""Detect and extract images/figures from PDF documents using PyMuPDF."""

import os

import fitz  # PyMuPDF
from loguru import logger

# Minimum dimensions (in pixels) to consider an image as a figure/chart
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100


def extract_visuals(pdf_path: str, output_dir: str) -> list[dict]:
    """Extract images from a PDF document and save them as files.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save extracted images.

    Returns:
        List of dicts with keys: page_number, image_path, bbox, image_index.
    """
    visuals: list[dict] = []
    os.makedirs(output_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Skip small images (logos, icons, etc.)
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue

                    # Save the image
                    image_filename = f"page_{page_num + 1}_img_{img_idx}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # Try to find the bounding box of this image on the page
                    bbox = _find_image_bbox(page, xref)

                    visuals.append(
                        {
                            "page_number": page_num + 1,
                            "image_path": image_path,
                            "bbox": bbox,
                            "image_index": img_idx,
                            "width": width,
                            "height": height,
                        }
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {img_idx} from page {page_num + 1}: {e}"
                    )
                    continue

        doc.close()
        logger.info(
            f"Extracted {len(visuals)} visual elements from {pdf_path}"
        )

    except Exception as e:
        logger.error(f"Failed to extract visuals from {pdf_path}: {e}")
        raise

    return visuals


def _find_image_bbox(page, xref: int) -> tuple | None:
    """Try to find the bounding box of an image on a page.

    A bounding box (bbox) is the rectangular region that defines where an image 
    sits on the PDF page. It's represented as four coordinates:
    (x0, y0, x1, y1)
    where;
    - (x0, y0) is the top-left corner
    - (x1, y1) is the bottom-right corner

    Returns (x0, y0, x1, y1) or None if not found.
    """
    try:
        for img in page.get_images(full=True):
            if img[0] == xref:
                # Get image rectangles
                rects = page.get_image_rects(img)
                if rects:
                    rect = rects[0]
                    return (rect.x0, rect.y0, rect.x1, rect.y1)
    except Exception:
        pass
    return None
