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
        grid_img = np.flipud(grid_img) # Flip vertically to match image coordinate system
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
        gh, gw = grid_shape
        fh, fw = floor_np.shape[:2]

        # center-to-center mapping + rotation/scale + translation
        cx_f, cy_f = fw / 2.0, fh / 2.0
        cx_g, cy_g = gw / 2.0, gh / 2.0

        M_rs = cv2.getRotationMatrix2D((cx_f, cy_f), rotation_deg, scale)
        M = np.vstack([M_rs, [0, 0, 1]]).astype(np.float32)

        T_center = np.array([[1, 0, (cx_g - cx_f)], [0, 1, (cy_g - cy_f)], [0, 0, 1]], dtype=np.float32)
        T_user = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

        M_full = T_user @ T_center @ M
        warped = cv2.warpAffine(
            floor_np,
            M_full[:2, :],
            (gw, gh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        print(f"[_apply_affine] Warped image shape: {warped.shape}")
        return warped
    
    def auto_align_with_orb(self, floor_img: Image.Image, grid_img: Image.Image) -> dict:
        """
        Use ORB feature matching + RANSAC to estimate affine transform.
        More reliable than LLM-only approach.
        """
        print("[auto_align_with_orb] Starting feature-based alignment...")
        
        # Convert to grayscale for feature detection
        floor_gray = cv2.cvtColor(np.array(floor_img), cv2.COLOR_RGB2GRAY)
        grid_gray = cv2.cvtColor(np.array(grid_img), cv2.COLOR_RGB2GRAY)
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, desc1 = orb.detectAndCompute(floor_gray, None)
        kp2, desc2 = orb.detectAndCompute(grid_gray, None)
        
        print(f"[auto_align_with_orb] Found {len(kp1)} floor keypoints, {len(kp2)} grid keypoints")
        
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test (Lowe's ratio)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"[auto_align_with_orb] {len(good_matches)} good matches after ratio test")
        
        if len(good_matches) < 10:
            print("[auto_align_with_orb] WARNING: Too few matches, falling back to LLM")
            return None
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate affine transform with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=2000
        )
        
        if M is None:
            print("[auto_align_with_orb] RANSAC failed, falling back to LLM")
            return None
        
        inlier_count = np.sum(inliers)
        print(f"[auto_align_with_orb] RANSAC: {inlier_count}/{len(good_matches)} inliers")
        
        # Extract scale, rotation, translation from affine matrix
        # M = [[a, b, tx], [c, d, ty]]
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        rotation_deg = np.degrees(np.arctan2(M[0, 1], M[0, 0]))
        tx = M[0, 2]
        ty = M[1, 2]
        
        params = {
            "scale": float(scale),
            "rotation_deg": float(rotation_deg),
            "tx": float(tx),
            "ty": float(ty),
            "alpha": 0.5,  # default blend
            "inliers": int(inlier_count),
            "affine_matrix": M.tolist()
        }
        
        print(f"[auto_align_with_orb] Estimated params: scale={scale:.3f}, rot={rotation_deg:.2f}°, tx={tx:.1f}, ty={ty:.1f}")
        return params
    
    def _apply_affine_direct(self, floor_img: Image.Image, grid_shape, affine_matrix: np.ndarray) -> np.ndarray:
        """Apply pre-computed affine matrix directly (from ORB/RANSAC)."""
        floor_np = np.array(floor_img)
        gh, gw = grid_shape
        
        M = np.array(affine_matrix, dtype=np.float32)
        warped = cv2.warpAffine(
            floor_np,
            M,
            (gw, gh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        print(f"[_apply_affine_direct] Warped shape: {warped.shape}")
        return warped

    def overlay_with_llm(
        self,
        floor_plan_path: str,
        output_path: str = "overlaid_map.png",
        use_orb: bool = True,
    ) -> np.ndarray:
        """
        Overlay floor plan onto occupancy grid.
        Tries ORB feature matching first, falls back to LLM if requested.
        """
        print(f"[overlay_with_llm] Loading floor plan: {floor_plan_path}")
        grid_img = self._grid_to_image(self.ogm.grid)
        floor_img = Image.open(floor_plan_path).convert("RGB")
        print(f"[overlay_with_llm] Floor: {floor_img.size}, Grid: {grid_img.size}")

        params = None
        if use_orb:
            params = self.auto_align_with_orb(floor_img, grid_img)
        
        # Fallback to LLM if ORB failed or disabled
        if params is None:
            print("[overlay_with_llm] Using LLM for alignment...")
            params = self._get_alignment_from_llm(grid_img, floor_img)
        
        alpha = float(params.get("alpha", 0.5))
        print(f"[overlay_with_llm] alpha={alpha}")

        # Apply transform
        if "affine_matrix" in params:
            warped_floor = self._apply_affine_direct(floor_img, self.ogm.grid.shape, params["affine_matrix"])
        else:
            warped_floor = self._apply_affine(floor_img, self.ogm.grid.shape, params)

        # Blend
        grid_np = np.array(grid_img)
        overlaid = cv2.addWeighted(grid_np, alpha, warped_floor, 1 - alpha, 0)

        # Save
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, overlaid)
        print(f"[overlay_with_llm] Saved: {output_path}")
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
    overlaid = overlay.overlay_with_llm("./src/elb_firstfloor_directorymap_cropped.png", "./output/overlaid_map_cropped.png")