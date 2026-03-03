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
        Includes validation to reject out-of-bounds results.
        """
        print("[auto_align_with_orb] Starting feature-based alignment...")
        
        # Convert to grayscale
        floor_gray = cv2.cvtColor(np.array(floor_img), cv2.COLOR_RGB2GRAY)
        grid_gray = cv2.cvtColor(np.array(grid_img), cv2.COLOR_RGB2GRAY)
        
        # Extract edges
        floor_edges = cv2.Canny(floor_gray, 50, 150)
        grid_edges = cv2.Canny(grid_gray, 30, 100)
        
        print(f"[auto_align_with_orb] Edge maps computed")
        
        # Detect ORB features on edges
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, desc1 = orb.detectAndCompute(floor_edges, None)
        kp2, desc2 = orb.detectAndCompute(grid_edges, None)
        
        print(f"[auto_align_with_orb] Found {len(kp1)} floor keypoints, {len(kp2)} grid keypoints")
        
        if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
            print("[auto_align_with_orb] WARNING: Insufficient keypoints on edge maps")
            return None
        
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"[auto_align_with_orb] {len(good_matches)} good matches after ratio test")
        
        if len(good_matches) < 10:
            print("[auto_align_with_orb] WARNING: Too few matches")
            return None
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate affine transform with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0, maxIters=5000
        )
        
        if M is None:
            print("[auto_align_with_orb] RANSAC failed")
            return None
        
        inlier_count = np.sum(inliers)
        print(f"[auto_align_with_orb] RANSAC: {inlier_count}/{len(good_matches)} inliers")
        
        # Validate transform (check if floor plan stays in bounds)
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        
        # Reject extremely small or large scales
        if scale < 0.1 or scale > 3.0:
            print(f"[auto_align_with_orb] Scale {scale:.3f} out of reasonable range [0.1, 3.0]")
            return None
        
        # Check if transformation moves image too far out of bounds
        fh, fw = np.array(floor_img).shape[:2]
        gh, gw = grid_img.size[1], grid_img.size[0]  # (width, height)
        
        # Transform corner points to check bounds
        corners = np.array([
            [0, 0, 1],
            [fw, 0, 1],
            [0, fh, 1],
            [fw, fh, 1]
        ], dtype=np.float32).T
        
        M_full = np.vstack([M, [0, 0, 1]])
        transformed_corners = M_full @ corners
        
        # Check if any corner is within reasonable bounds
        in_bounds = np.sum((transformed_corners[0, :] >= -fw) & (transformed_corners[0, :] <= gw + fw) &
                           (transformed_corners[1, :] >= -fh) & (transformed_corners[1, :] <= gh + fh))
        
        if in_bounds < 2:
            print(f"[auto_align_with_orb] Warning: transformed image mostly out of bounds")
            return None
        
        rotation_deg = np.degrees(np.arctan2(M[0, 1], M[0, 0]))
        tx = M[0, 2]
        ty = M[1, 2]
        
        params = {
            "scale": float(scale),
            "rotation_deg": float(rotation_deg),
            "tx": float(tx),
            "ty": float(ty),
            "alpha": 0.5,
            "inliers": int(inlier_count),
            "affine_matrix": M.tolist()
        }
        
        print(f"[auto_align_with_orb] Valid params: scale={scale:.3f}, rot={rotation_deg:.2f}°, tx={tx:.1f}, ty={ty:.1f}")
        return params
    
    def auto_align_with_ecc(self, floor_img: Image.Image, grid_img: Image.Image) -> dict:
        """
        Use ECC (Enhanced Correlation Coefficient) for dense image alignment.
        Works better than ORB for cross-modal images.
        """
        print("[auto_align_with_ecc] Starting ECC-based alignment...")
        
        floor_gray = cv2.cvtColor(np.array(floor_img), cv2.COLOR_RGB2GRAY)
        grid_gray = cv2.cvtColor(np.array(grid_img), cv2.COLOR_RGB2GRAY)
        
        # Resize to smaller size for faster computation
        scale_factor = 0.5
        floor_small = cv2.resize(floor_gray, None, fx=scale_factor, fy=scale_factor)
        grid_small = cv2.resize(grid_gray, None, fx=scale_factor, fy=scale_factor)
        
        print(f"[auto_align_with_ecc] Resized to {floor_small.shape}")
        
        # Define the warp mode (affine = 2D translation + rotation + scale)
        warp_mode = cv2.MOTION_AFFINE
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-3)
        
        # Initialize identity matrix
        if warp_mode == cv2.MOTION_AFFINE:
            M = np.eye(2, 3, dtype=np.float32)
        
        try:
            cc, M = cv2.findTransformECC(grid_small, floor_small, M, warp_mode, criteria)
            print(f"[auto_align_with_ecc] ECC correlation: {cc:.4f}")
        except Exception as e:
            print(f"[auto_align_with_ecc] ECC failed: {e}")
            return None
        
        # Scale back to original image size
        M[0, 2] /= scale_factor
        M[1, 2] /= scale_factor
        
        # Extract parameters
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        rotation_deg = np.degrees(np.arctan2(M[0, 1], M[0, 0]))
        tx = M[0, 2]
        ty = M[1, 2]
        
        params = {
            "scale": float(scale),
            "rotation_deg": float(rotation_deg),
            "tx": float(tx),
            "ty": float(ty),
            "alpha": 0.5,
            "affine_matrix": M.tolist(),
            "ecc_score": float(cc)
        }
        
        print(f"[auto_align_with_ecc] Estimated: scale={scale:.3f}, rot={rotation_deg:.2f}°, tx={tx:.1f}, ty={ty:.1f}")
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

    def manual_alignment_interactive(self, floor_img: Image.Image, grid_img: Image.Image) -> dict:
        """
        Interactive manual alignment: user clicks 4-6 corresponding points.
        Uses least-squares to fit affine (more robust than 3-point exact).
        """
        print("[manual_alignment_interactive] Starting manual alignment...")
        print("Instructions:")
        print("1. Window shows FLOOR PLAN")
        print("2. Click 4-6 corresponding landmark points (corners, doorways, centers)")
        print("3. Then click SAME points in the OCCUPANCY GRID (same order!)")
        print("4. Press SPACE to finish, ESC to cancel\n")
        
        floor_np = np.array(floor_img)
        grid_np = np.array(grid_img)
        
        floor_points = []
        grid_points = []
        
        def mouse_callback_floor(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                floor_points.append([x, y])
                print(f"Floor point {len(floor_points)}: ({x}, {y})")
                cv2.circle(display_floor, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(display_floor, str(len(floor_points)), (x+10, y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Floor Plan - Click Points (SPACE=done, ESC=cancel)", display_floor)
        
        def mouse_callback_grid(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                grid_points.append([x, y])
                print(f"Grid point {len(grid_points)}: ({x}, {y})")
                cv2.circle(display_grid, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(display_grid, str(len(grid_points)), (x+10, y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Occupancy Grid - Click Points (SPACE=done, ESC=cancel)", display_grid)
        
        # Step 1: Click points on floor plan
        display_floor = floor_np.copy()
        cv2.namedWindow("Floor Plan - Click Points (SPACE=done, ESC=cancel)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Floor Plan - Click Points (SPACE=done, ESC=cancel)", 800, 600)
        cv2.setMouseCallback("Floor Plan - Click Points (SPACE=done, ESC=cancel)", mouse_callback_floor)
        cv2.imshow("Floor Plan - Click Points (SPACE=done, ESC=cancel)", display_floor)
        
        while len(floor_points) < 4:
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                print("[manual_alignment_interactive] Cancelled")
                cv2.destroyAllWindows()
                return None
            elif key == 32:  # SPACE
                if len(floor_points) >= 4:
                    break
                else:
                    print(f"Need at least 4 points. Current: {len(floor_points)}")
        
        cv2.destroyAllWindows()
        print(f"[manual_alignment_interactive] Collected {len(floor_points)} floor points")
        
        # Step 2: Click points on occupancy grid
        display_grid = grid_np.copy()
        cv2.namedWindow("Occupancy Grid - Click Points (SPACE=done, ESC=cancel)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Occupancy Grid - Click Points (SPACE=done, ESC=cancel)", 800, 600)
        cv2.setMouseCallback("Occupancy Grid - Click Points (SPACE=done, ESC=cancel)", mouse_callback_grid)
        cv2.imshow("Occupancy Grid - Click Points (SPACE=done, ESC=cancel)", display_grid)
        
        while len(grid_points) < len(floor_points):
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                print("[manual_alignment_interactive] Cancelled")
                cv2.destroyAllWindows()
                return None
            elif key == 32:  # SPACE
                if len(grid_points) >= len(floor_points):
                    break
                else:
                    print(f"Need {len(floor_points)} points. Current: {len(grid_points)}")
        
        cv2.destroyAllWindows()
        print(f"[manual_alignment_interactive] Collected {len(grid_points)} grid points")
        
        # Compute affine using least-squares fit (more robust)
        src_pts = np.float32(floor_points)
        dst_pts = np.float32(grid_points)
        
        # Use first 3 points for exact affine, but validate with remaining points
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
        
        # Optionally: fit to all points using least-squares
        # Build overdetermined system: [x y 1] * [a b tx; c d ty]^T = [x' y']
        A = np.hstack([src_pts, np.ones((len(src_pts), 1))])  # shape: (N, 3)
        # Solve for each row of M separately
        M_ls_row0, _, _, _ = np.linalg.lstsq(A, dst_pts[:, 0], rcond=None)
        M_ls_row1, _, _, _ = np.linalg.lstsq(A, dst_pts[:, 1], rcond=None)
        M_ls = np.vstack([M_ls_row0, M_ls_row1])
        
        # Calculate reprojection error with least-squares fit
        transformed = (A @ M_ls.T)
        error = np.mean(np.sqrt(np.sum((transformed - dst_pts)**2, axis=1)))
        print(f"[manual_alignment_interactive] Least-squares reprojection error: {error:.2f} pixels")
        
        # Use least-squares if error is reasonable
        if error < 20:
            M = M_ls
            print(f"[manual_alignment_interactive] Using least-squares fit (error: {error:.2f})")
        else:
            print(f"[manual_alignment_interactive] Error {error:.2f} high, using 3-point fit instead")
        
        print(f"[manual_alignment_interactive] Computed affine matrix:\n{M}")
        
        # Extract parameters
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        rotation_deg = np.degrees(np.arctan2(M[0, 1], M[0, 0]))
        tx = M[0, 2]
        ty = M[1, 2]
        
        params = {
            "scale": float(scale),
            "rotation_deg": float(rotation_deg),
            "tx": float(tx),
            "ty": float(ty),
            "alpha": 0.5,
            "affine_matrix": M.tolist(),
            "method": "manual_points_ls",
            "error": float(error)
        }
        
        print(f"[manual_alignment_interactive] Extracted: scale={scale:.3f}, rot={rotation_deg:.2f}°, tx={tx:.1f}, ty={ty:.1f}")
        return params
    
    def overlay_with_llm(
        self,
        floor_plan_path: str,
        output_path: str = "overlaid_map.png",
        use_manual: bool = False,
        use_ecc: bool = True,
        use_orb: bool = False,
    ) -> np.ndarray:
        """
        Overlay floor plan onto occupancy grid.
        Tries: Manual -> ECC -> ORB -> LLM (based on flags).
        """
        print(f"[overlay_with_llm] Loading floor plan: {floor_plan_path}")
        grid_img = self._grid_to_image(self.ogm.grid)
        floor_img = Image.open(floor_plan_path).convert("RGB")
        print(f"[overlay_with_llm] Floor: {floor_img.size}, Grid: {grid_img.size}")

        params = None
        
        # Try manual alignment first
        if use_manual:
            print("[overlay_with_llm] Attempting manual alignment...")
            params = self.manual_alignment_interactive(floor_img, grid_img)
        
        # Try ECC
        if params is None and use_ecc:
            print("[overlay_with_llm] Attempting ECC alignment...")
            params = self.auto_align_with_ecc(floor_img, grid_img)
        
        # Try ORB
        if params is None and use_orb:
            print("[overlay_with_llm] Attempting ORB alignment...")
            params = self.auto_align_with_orb(floor_img, grid_img)
        
        # Final fallback to LLM
        if params is None:
            print("[overlay_with_llm] Using LLM for alignment...")
            params = self._get_alignment_from_llm(grid_img, floor_img)
        
        alpha = float(params.get("alpha", 0.5))
        print(f"[overlay_with_llm] alpha={alpha}, method={params.get('method', 'unknown')}")

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

        success = cv2.imwrite(output_path, overlaid)
        if success:
            print(f"[overlay_with_llm] Saved: {output_path}")
        else:
            print(f"[overlay_with_llm] ERROR saving to {output_path}")
        
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
    overlaid = overlay.overlay_with_llm(
        "./src/elb_firstfloor_directorymap_cropped.png", 
        "./output/overlaid_map_cropped_manual.png",
        use_manual=True,
        use_ecc=False,
        use_orb=False,
    )