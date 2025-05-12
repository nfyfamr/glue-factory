from ...geometry.gt_generation import (
    gt_line_matches_from_homography,
    gt_matches_from_homography,
    gt_warp_from_homography,
)
from ..base_model import BaseModel


class HomographyMatcher(BaseModel):
    default_conf = {
        # GT parameters for points
        "use_points": True,
        "th_positive": 3.0,
        "th_negative": 3.0,
        # GT parameters for lines
        "use_lines": False,
        "n_line_sampled_pts": 50,
        "line_perp_dist_th": 5,
        "overlap_th": 0.2,
        "min_visibility_th": 0.5,
        # GT parameters for flow
        "use_warp": False,
        # GT parameters for init matching
        "use_init_matching": False,
        "init_th_positive": 8.0,
        "init_th_negative": 8.0,
        "init_n_line_sampled_pts": 50,
        "init_line_perp_dist_th": 5,
        "init_overlap_th": 0.2,
        "init_min_visibility_th": 0.5,
    }

    required_data_keys = ["H_0to1"]

    def _init(self, conf):
        # TODO (iago): Is this just boilerplate code?
        if self.conf.use_points:
            self.required_data_keys += ["keypoints0", "keypoints1"]
            if self.conf.use_init_matching:
                self.required_data_keys += ["init_keypoints0", "init_keypoints1"]
        if self.conf.use_lines:
            self.required_data_keys += [
                "lines0",
                "lines1",
                "valid_lines0",
                "valid_lines1",
            ]
        if self.conf.use_warp:
            self.required_data_keys += ["intermediate_ksamples0"]

    def _forward(self, data):
        result = {}
        if self.conf.use_points:
            result = gt_matches_from_homography(
                data["keypoints0"],
                data["keypoints1"],
                data["H_0to1"],
                pos_th=self.conf.th_positive,
                neg_th=self.conf.th_negative,
            )
        if self.conf.use_lines:
            line_assignment, line_m0, line_m1 = gt_line_matches_from_homography(
                data["lines0"],
                data["lines1"],
                data["valid_lines0"],
                data["valid_lines1"],
                data["view0"]["image"].shape,
                data["view1"]["image"].shape,
                data["H_0to1"],
                self.conf.n_line_sampled_pts,
                self.conf.line_perp_dist_th,
                self.conf.overlap_th,
                self.conf.min_visibility_th,
            )
            result["line_matches0"] = line_m0
            result["line_matches1"] = line_m1
            result["line_assignment"] = line_assignment
        if self.conf.use_warp:
            if not self.training and data["intermediate_ksamples0"] is None:
                result["patch_warps0_1"] = None
                result["patch_warps0_1_prob"] = None
            else:
                warp, warp_prob = gt_warp_from_homography(
                    data["intermediate_ksamples0"],
                    data["view1"]["image"].shape[-2:],
                    data["H_0to1"],
                    inverse=False,
                )
                result["patch_warps0_1"] = warp
                result["patch_warps0_1_prob"] = warp_prob
        if self.conf.use_init_matching:
            if self.conf.use_points:
                result["init"] = gt_matches_from_homography(
                    data["init_keypoints0"],
                    data["init_keypoints1"],
                    data["H_0to1"],
                    pos_th=self.conf.init_th_positive,
                    neg_th=self.conf.init_th_negative,
                )
        return result

    def loss(self, pred, data):
        raise NotImplementedError
