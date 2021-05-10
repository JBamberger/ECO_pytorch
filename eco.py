import math

import torch

from .lib import complex, dcf, fourier
from .lib.dcf import RegWindowParams
from .lib.optimization import GaussNewtonCG, FactorizedConvProblem, FilterOptim
from .lib.tensorlist import TensorList, tensor_operation
from .params import ECOParams


class ECO:
    """
    Partial ECO implementation with missing generative subspace model. The Implementation only handles a single feature
    resolution. The extension with hand-crafted features is not available.
    """
    params: ECOParams

    def __init__(self, params):
        self.params = params

    def initialize(self, image, init_box):
        state = init_box  # format: xywh
        self.frame_num = 1

        # Get position and size
        self.pos = torch.Tensor([
            state[1] + (state[3] - 1) / 2,
            state[0] + (state[2] - 1) / 2
        ])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        if search_area > self.params.max_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.min_image_sample_size)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        self.img_sample_sz = torch.round(
            torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        self.img_sample_sz += 1 - self.img_sample_sz % 2

        # Set other sizes (corresponds to ECO code)
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.filter_sz = self.feature_sz + (self.feature_sz + 1) % 2
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
        self.compressed_dim = 16

        # Number of filters
        self.num_filters = len(self.filter_sz)

        # Get window function
        self.window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])

        # Get interpolation function
        self.interp_fs = TensorList([dcf.get_interp_fourier(sz, self.params.interpolation_bicubic_a, self.params.device)
                                     for sz in self.filter_sz])

        # Get regularization filter
        self.reg_filter = TensorList([
            dcf.get_reg_filter(self.img_support_sz, self.base_target_sz, RegWindowParams()).to(self.params.device)])
        self.reg_energy = self.reg_filter.view(-1) @ self.reg_filter.view(-1)

        # Get label function
        output_sigma_factor = 1 / 4
        sigma = (self.filter_sz / self.img_support_sz) * \
                torch.sqrt(self.base_target_sz.prod()) * \
                output_sigma_factor
        self.yf = TensorList(
            [dcf.label_function(sz, sig).to(self.params.device) for sz, sig in zip(self.filter_sz, sigma)])

        # Optimization options
        self.params.precond_learning_rate = TensorList([0.0075])
        self.params.direction_forget_factor = (1 - max(
            self.params.precond_learning_rate)) ** self.params.CG_forgetting_rate

        # Convert image
        im = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

        # Setup bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample

        # list, shape [1, 32, 237, 237]
        x = self.params.features.extract_transformed(im, self.pos, self.target_scale, self.img_sample_sz)
        assert len(x) == 1

        # convert to [D, N*H*W], feature channels in the front
        # Initialize projection matrix
        self.projection_matrix = TensorList([self.compute_projection_matrix(x[0])])

        # Transform to get the training sample
        # list of shape [1, 32, 237, 119, 2]
        train_xf = self.preprocess_sample(x)

        # Shift sample
        shift_samp = 2 * math.pi * (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Initialize first-frame training samples
        num_init_samples = train_xf.size(0)
        init_sample_weights = TensorList([xf.new_ones(1) / xf.shape[0] for xf in train_xf])
        init_training_samples = train_xf.permute(2, 3, 0, 1, 4)

        # Sample counters and weights
        self.num_stored_samples = num_init_samples
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([xf.new_zeros(self.params.sample_memory_size) for xf in train_xf])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, num_init_samples):
            sw[:num] = init_sw

        # Initialize memory and filter
        assert len(train_xf) == 1
        xf = train_xf[0]
        self.training_samples = TensorList(
            [xf.new_zeros(xf.shape[2], xf.shape[3], self.params.sample_memory_size, self.compressed_dim, 2)])
        self.filter = TensorList(
            [xf.new_zeros(1, self.compressed_dim, xf.shape[2], xf.shape[3], 2)])

        # Do joint optimization
        joint_problem = FactorizedConvProblem(
            init_training_samples, self.yf, self.reg_filter, self.projection_matrix, self.params, init_sample_weights)
        joint_var = self.filter.concat(self.projection_matrix)
        joint_optimizer = GaussNewtonCG(joint_problem, joint_var)

        if self.params.update_projection_matrix:
            joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)

        # Re-project samples with the new projection matrix
        compressed_samples = complex.mtimes(init_training_samples, self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:, :, :init_samp.shape[2], :, :] = init_samp

        # Initialize optimizer
        self.filter_optimizer = FilterOptim(self.params, self.reg_energy)
        self.filter_optimizer.register(
            self.filter, self.training_samples, self.yf, self.sample_weights, self.reg_filter)
        self.filter_optimizer.sample_energy = joint_problem.sample_energy

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)

        self.symmetrize_filter()

    def compute_projection_matrix(self, e: torch.Tensor):
        """
        Computes the initial projection matrix.
        :param e: sample of shape [1, 32, 237, 237]
        :return:
        """
        x_mat = e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).clone()
        x_mat -= x_mat.mean(dim=1, keepdim=True)
        cov_x = x_mat @ x_mat.t()
        return torch.svd(cov_x)[0][:, :self.compressed_dim].clone()

    def track(self, image) -> dict:
        self.frame_num += 1

        # Convert image
        im = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors
        test_xf = self.extract_fourier_sample(im, self.pos, sample_scales, self.img_sample_sz)

        # Compute scores
        sf = complex.mult(self.filter, test_xf).sum(1, keepdim=True)
        translation_vec, scale_ind, s = self.localize_target(sf)
        scale_change_factor = self.params.scale_factors[scale_ind]

        # Update position and scale
        self.update_state(sample_pos + translation_vec, self.target_scale * scale_change_factor)

        # ------- UPDATE ------- #

        # Get train sample
        train_xf = TensorList([xf[scale_ind:scale_ind + 1, ...] for xf in test_xf])

        # Shift the sample
        shift_samp = 2 * math.pi * (self.pos - sample_pos) / (sample_scales[scale_ind] * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Update memory
        self.update_memory(train_xf)

        # Train filter
        if self.frame_num % self.params.train_skipping == 1:
            self.filter_optimizer.run(self.params.CG_iter, train_xf)
            self.symmetrize_filter()

        # Return new state
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        return {'target_bbox': new_state.tolist()}

    def localize_target(self, sf: TensorList):
        weight = 0.6
        scores = fourier.sample_fs(fourier.sum_fs(weight * sf), self.output_sz)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        translation_vec *= self.params.scale_factors[scale_ind]

        return translation_vec, scale_ind, scores

    def extract_fourier_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> TensorList:
        x = self.params.features.extract(im, pos, scales, sz)[0]

        @tensor_operation
        def _project_sample(x: torch.Tensor, P: torch.Tensor):
            if P is None:
                return x

            # [N,D,H,W] -> [H,W,N,D] * [H,W,N,C] -> [H,W,N,C] -> [N,C,H,W]
            return torch.matmul(x.permute(2, 3, 0, 1), P).permute(2, 3, 0, 1)

        projected_x = _project_sample(x, self.projection_matrix)

        return self.preprocess_sample(projected_x)

    def preprocess_sample(self, x: TensorList) -> TensorList:
        x *= self.window
        sample_xf = fourier.cfft2(x)
        return TensorList([dcf.interpolate_dft(xf, bf) for xf, bf in zip(sample_xf, self.interp_fs)])

    def update_memory(self, sample_xf: TensorList):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp in zip(self.sample_weights, self.previous_replace_ind, self.num_stored_samples):
            if num_samp == 0:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw, 0)
                r_ind = r_ind.item()

                # Update weights
                if prev_ind is None:
                    sw /= 1 - 0.0075
                    sw[r_ind] = 0.0075
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - 0.0075)

            sw /= sw.sum()
            replace_ind.append(r_ind)

        self.previous_replace_ind = replace_ind.copy()
        self.num_stored_samples += 1
        for train_samp, xf, ind in zip(self.training_samples, sample_xf, replace_ind):
            train_samp[:, :, ind:ind + 1, :, :] = xf.permute(2, 3, 0, 1, 4)

    def update_state(self, new_pos, new_scale):
        # Update scale
        self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
        self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def symmetrize_filter(self):
        """
        Ensures hermitian symmetry of each filter.
        :return:
        """

        for hf in self.filter:
            hf[:, :, :, 0, :] /= 2
            hf[:, :, :, 0, :] += complex.conj(hf[:, :, :, 0, :].flip((2,)))
