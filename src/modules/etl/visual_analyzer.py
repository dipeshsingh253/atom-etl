"""Analyze extracted images using a vision AI provider to extract structured data."""

from loguru import logger

from src.providers.base import BaseVisionProvider

CHART_EXTRACTION_PROMPT = """Analyze this image from a PDF document.

If this image contains a chart, graph, heatmap, or any data visualization:
1. Identify the type of visualization (bar chart, pie chart, line graph, heatmap, etc.)
2. Extract ALL numerical values and their labels

Return a JSON object in this exact format:
{
    "is_data_visualization": true,
    "visual_type": "<type of chart>",
    "title": "<title of the chart if visible>",
    "data": [
        {"label": "<label>", "value": "<numerical value>"}
    ]
}

If this image is NOT a data visualization (e.g., it's a logo, photograph, decorative image):
{
    "is_data_visualization": false,
    "visual_type": "non_data",
    "title": "",
    "data": []
}

Important:
- Extract ALL data points visible in the chart.
- Use exact numbers as shown (include units if present).
- For heatmaps, include both the row label and column label in the "label" field.
- Return ONLY valid JSON, no other text."""


async def analyze_visual(
    image_path: str,
    vision_provider: BaseVisionProvider,
) -> dict:
    """Analyze a single image and extract structured data.

    Args:
        image_path: Path to the image file.
        vision_provider: Vision provider instance.

    Returns:
        Dict with keys: is_data_visualization, visual_type, title, data.
    """
    try:
        # TODO: Before this step we can add a quick pre-filter using a smaller model to classify if the image is likely to contain data or not. This can save costs by only sending likely candidates to the more expensive vision provider for detailed analysis.
        result = await vision_provider.extract_structured_data(
            image_path=image_path,
            prompt=CHART_EXTRACTION_PROMPT,
        )

        # Handle parse errors from the provider
        if result.get("parse_error"):
            logger.warning(f"Could not parse structured data from {image_path}")
            return {
                "is_data_visualization": False,
                "visual_type": "unknown",
                "title": "",
                "data": [],
                "raw_response": result.get("raw_response", ""),
            }

        return result

    except Exception as e:
        logger.error(f"Vision analysis failed for {image_path}: {e}")
        return {
            "is_data_visualization": False,
            "visual_type": "error",
            "title": "",
            "data": [],
            "error": str(e),
        }


async def analyze_visuals_batch(
    visuals: list[dict],
    vision_provider: BaseVisionProvider,
) -> list[dict]:
    """Analyze a batch of extracted images.

    Args:
        visuals: List of dicts from visual_extractor (with image_path, page_number, etc.)
        vision_provider: Vision provider instance.

    Returns:
        List of dicts combining visual metadata with analysis results.
    """
    results: list[dict] = []

    for visual in visuals:
        image_path = visual["image_path"]
        page_number = visual["page_number"]

        logger.info(f"Analyzing visual from page {page_number}: {image_path}")
        analysis = await analyze_visual(image_path, vision_provider)

        results.append(
            {
                "page_number": page_number,
                "image_path": image_path,
                **analysis,
            }
        )

    data_visuals = [r for r in results if r.get("is_data_visualization")]
    logger.info(
        f"Analyzed {len(results)} visuals, {len(data_visuals)} contain data"
    )

    return results
