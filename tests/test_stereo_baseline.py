import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import stereo_baseline as sb


SIZE = (1920, 1080)
K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
PATTERN = (6, 9)


def look_at(position, target, up=(0.0, -1.0, 0.0)):
    """World->camera rotation and translation for a camera at position looking at target (y down)."""
    z = np.asarray(target, float) - position
    z /= np.linalg.norm(z)
    x = np.cross(-np.asarray(up), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ position


def synthetic_views(n_views=20, noise_px=0.1, seed=0):
    """Two cameras ~4.8 m apart (13 cm height difference) watching a hand-held 7x10-square board."""
    rng = np.random.default_rng(seed)
    obj = sb.board_object_points(PATTERN, sb.SQUARE_MM)
    centre = obj.mean(axis=0)
    cam1 = np.array([0.0, 0.0, 0.0])
    cam2 = np.array([3400.0, -130.0, 3400.0])  # y is down, so camera 2 is 13 cm higher
    target = np.array([3000.0, 0.0, 400.0])  # near the room corner: the cameras look at it ~90° apart
    cams = [look_at(cam1, target), look_at(cam2, target)]
    view_dir = (cam1 + cam2) / 2 - target  # the board faces between the cameras
    pts1, pts2 = [], []
    while len(pts1) < n_views:
        # board pose in the world: face the cameras, random tilt, random position near the target
        normal = view_dir / np.linalg.norm(view_dir) + rng.normal(0, 0.25, 3)
        R_b, _ = look_at(np.zeros(3), -normal, up=(rng.normal(0, 0.3), -1.0, 0))
        R_wb = R_b.T  # board axes in world
        t_wb = target + rng.normal(0, 250, 3) - R_wb @ centre
        world = (R_wb @ obj.T).T + t_wb
        projected, visible = [], True
        for R_c, t_c in cams:
            pc = (R_c @ world.T).T + t_c
            # board in front of the camera, front face towards it (OpenCV: board z points away from the camera)
            facing = (R_c @ R_wb[:, 2]) @ (pc.mean(axis=0) / np.linalg.norm(pc.mean(axis=0)))
            p = (K @ pc.T).T
            p = p[:, :2] / p[:, 2:]
            if np.any(pc[:, 2] <= 0) or facing < 0.3 or np.any(p < 0) or np.any(p >= SIZE):
                visible = False
            projected.append((p + rng.normal(0, noise_px, p.shape)).astype(np.float32))
        if visible:
            pts1.append(projected[0])
            pts2.append(projected[1])
    return obj, pts1, pts2, np.linalg.norm(cam2 - cam1)


def render_board(squares=(7, 10), px=40, margin=60):
    """Grey image of a checkerboard (top-left square dark), and the homography-free corner grid."""
    cols, rows = squares
    img = np.full((rows * px + 2 * margin, cols * px + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[margin + r * px:margin + (r + 1) * px, margin + c * px:margin + (c + 1) * px] = 0
    return img


class StereoBaselineTests(unittest.TestCase):
    def test_stereo_recovers_known_baseline(self):
        obj, pts1, pts2, truth_mm = synthetic_views()
        intr = {cam: sb.Intrinsics("standard", K, np.zeros(5), "synthetic") for cam in sb.CAMERAS}
        paired = [(f"#{i}", a, b) for i, (a, b) in enumerate(zip(pts1, pts2))]
        result = sb.estimate(obj, intr, paired, SIZE, height_delta_cm=13.0, unit_to_cm=0.1)
        truth_cm = truth_mm / 10
        self.assertAlmostEqual(result["stereo"]["baseline_3d_cm"], truth_cm, delta=0.01 * truth_cm)
        self.assertLess(result["stereo"]["rms_px"], 0.5)
        self.assertAlmostEqual(result["pnp"]["baseline_3d_cm_median"], truth_cm, delta=0.01 * truth_cm)
        floor = math.sqrt(truth_cm ** 2 - 13.0 ** 2)
        self.assertAlmostEqual(result["stereo"]["floor_distance_cm"], floor, delta=0.01 * floor)

    def test_single_pair_pnp_fallback(self):
        obj, pts1, pts2, truth_mm = synthetic_views(n_views=1, noise_px=0.05, seed=3)
        R, T = sb.pnp_relative_pose(obj, pts1[0], pts2[0], K, K)
        self.assertAlmostEqual(np.linalg.norm(T), truth_mm, delta=0.03 * truth_mm)

    def test_fisheye_points_are_undistorted_before_stereo(self):
        obj, pts1, pts2, truth_mm = synthetic_views(seed=1)
        D = np.array([0.1, -0.05, 0.02, -0.01])
        fisheye = sb.Intrinsics("fisheye", K, D, "synthetic")
        distorted = [cv2.fisheye.distortPoints(cv2.undistortPoints(p.reshape(-1, 1, 2), K, None), K, D)
                     .reshape(-1, 2) for p in pts1]
        intr = {"front": fisheye, "side": sb.Intrinsics("standard", K, np.zeros(5), "synthetic")}
        paired = [(f"#{i}", a, b) for i, (a, b) in enumerate(zip(distorted, pts2))]
        result = sb.estimate(obj, intr, paired, SIZE, height_delta_cm=13.0, unit_to_cm=0.1)
        self.assertAlmostEqual(result["stereo"]["baseline_3d_cm"], truth_mm / 10, delta=truth_mm / 1000)

    def test_detection_order_is_the_same_when_the_board_is_upside_down(self):
        img = render_board()
        upright, _ = sb.detect_board(img, PATTERN)
        flipped, _ = sb.detect_board(cv2.rotate(img, cv2.ROTATE_180), PATTERN)
        self.assertIsNotNone(upright)
        self.assertIsNotNone(flipped)
        h, w = img.shape
        flipped_back = np.column_stack([w - 1 - flipped[:, 0], h - 1 - flipped[:, 1]])
        np.testing.assert_allclose(flipped_back, upright, atol=0.5)

    def test_probe_picks_matching_pattern(self):
        img = render_board()
        pattern, hits = sb.probe_pattern([("a", img, img)], sb.CANDIDATE_PATTERNS, max_frames=5)
        self.assertEqual(pattern, (6, 9))
        self.assertEqual(hits["7x10"], 0)

    def test_room_check(self):
        check = sb.room_check(480.0, 380.0, 336.0, 30.0)
        self.assertTrue(check["in_range"])
        self.assertEqual(check["expected_floor_range_cm"], [464.9, 507.2])
        d = check["equal_offsets_cm"]
        self.assertAlmostEqual(math.hypot(380 - d, 336 - d), 480.0, delta=0.2)
        self.assertFalse(sb.room_check(430.0, 380.0, 336.0, 30.0)["in_range"])
        self.assertTrue(math.isnan(sb.floor_distance(10.0, 13.0)))


if __name__ == "__main__":
    unittest.main()
