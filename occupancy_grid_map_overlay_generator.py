import os
import json
import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from openai import AzureOpenAI
from occupancy_grid_map_generator import OccupancyGridMap

load_dotenv()  # Load environment variables from .env file


class OccupancyMapOverlay:
    def __init__(self, occupancy_grid_map: OccupancyGridMap):
        """
        Initialize overlay helper with Azure OpenAI connection.
        """
        print("[OccupancyMapOverlay] Initializing...")
        self.ogm = occupancy_grid_map
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            default_headers={"OpenAI-Organization": os.getenv("OPENAI_ORGANIZATION")},
        )
        self.model = os.getenv("MODEL")
        print(f"[OccupancyMapOverlay] Model: {self.model}")
        print(f"[OccupancyMapOverlay] Grid shape: {self.ogm.grid.shape}")

    def _grid_to_image(self, grid: np.ndarray) -> Image.Image:
        """Convert occupancy grid to a visible image (free=white, occupied=black)."""
        print(f"[_grid_to_image] Converting grid to image. Grid shape: {grid.shape}")
        grid_img = (1 - grid) * 255
        grid_img = grid_img.astype(np.uint8)
        return Image.fromarray(grid_img).convert("RGB")

    def _image_to_base64(self, img: Image.Image) -> str:
        """Encode PIL image to base64 data URL."""
        print(f"[_image_to_base64] Encoding image. Size: {img.size}")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _get_alignment_from_llm(self, grid_img: Image.Image, floor_img: Image.Image) -> dict:
        """
        Ask multimodal LLM to estimate alignment parameters for the floorplan
        onto the occupancy grid. Returns JSON with scale, rotation_deg, tx, ty, alpha.
        """
        print("[_get_alignment_from_llm] Preparing images for LLM...")
        grid_b64 = self._image_to_base64(grid_img)
        floor_b64 = self._image_to_base64(floor_img)

        prompt = (
            "You are aligning a floorplan image onto an occupancy grid map.\n"
            "Estimate a 2D affine transform that maps the floorplan to the grid.\n"
            "Return ONLY JSON with fields:\n"
            "{"
            "\"scale\": float, "
            "\"rotation_deg\": float, "
            "\"tx\": float, "
            "\"ty\": float, "
            "\"alpha\": float"
            "}\n"
            "Where tx, ty are pixel translations in the grid image coordinate frame.\n"
            "alpha is blend weight for the grid (0..1)."
        )

        print("[_get_alignment_from_llm] Sending request to Azure OpenAI...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in image alignment for robotics."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": grid_b64}},
                        {"type": "image_url", "image_url": {"url": floor_b64}},
                    ],
                },
            ],
        )

        text = response.choices[0].message.content.strip()
        print(f"[_get_alignment_from_llm] Raw response: {text}")
        params = json.loads(text)
        print(f"[_get_alignment_from_llm] Parsed parameters: {params}")
        return params

    def _apply_affine(self, floor_img: Image.Image, grid_shape, params: dict) -> np.ndarray:
        """Apply affine transform to floorplan image."""
        scale = float(params.get("scale", 1.0))
        rotation_deg = float(params.get("rotation_deg", 0.0))
        tx = float(params.get("tx", 0.0))
        ty = float(params.get("ty", 0.0))
        print(f"[_apply_affine] scale={scale}, rotation_deg={rotation_deg}, tx={tx}, ty={ty}")

        floor_np = np.array(floor_img)
        h, w = grid_shape

        # Compose affine transform: scale + rotation around center, then translate
        center = (floor_np.shape[1] / 2, floor_np.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, rotation_deg, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        warped = cv2.warpAffine(
            floor_np,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        print(f"[_apply_affine] Warped image shape: {warped.shape}")
        return warped

    def overlay_with_llm(
        self,
        floor_plan_path: str,
        output_path: str = "overlaid_map.png",
    ) -> np.ndarray:
        """
        Use multimodal GPT‑5 to estimate alignment, then overlay.
        """
        print(f"[overlay_with_llm] Loading floor plan: {floor_plan_path}")
        grid_img = self._grid_to_image(self.ogm.grid)
        floor_img = Image.open(floor_plan_path).convert("RGB")
        print(f"[overlay_with_llm] Floor image size: {floor_img.size}")

        params = self._get_alignment_from_llm(grid_img, floor_img)
        alpha = float(params.get("alpha", 0.5))
        print(f"[overlay_with_llm] alpha={alpha}")

        warped_floor = self._apply_affine(floor_img, self.ogm.grid.shape, params)

        # Prepare grid image for blending
        grid_np = np.array(grid_img)
        overlaid = cv2.addWeighted(grid_np, alpha, warped_floor, 1 - alpha, 0)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        cv2.imwrite(output_path, overlaid)
        print(f"[overlay_with_llm] Saved output: {output_path}")
        return overlaid


# Usage example
if __name__ == "__main__":
    # Load occupancy grid map from Omron .map file
    ogm = OccupancyGridMap.from_omron_map(
        "./src/Base_Maps/input/input.map",
        occ_grid_res_mm=100,
        padding_mm=600,
        enable_padding=True,
    )
    ogm.visualize()

    # Create overlay generator and produce overlaid image
    overlay = OccupancyMapOverlay(ogm)
    overlaid = overlay.overlay_with_llm("./src/elb_firstfloor_directorymap.png", "./output/overlaid_map.png")