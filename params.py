import torch

from .lib.features import MultiResolutionExtractor


class ECOParams:
    """
    lambda == 2 * 10 ** -7
    first frame:
        10 GN iterations
        20 CG iterations
    Initial filter is zero
    P0 initialized by PCA, random is equally robust

    sample space model:
        gamma = 0.012
        num components L = 50
        num samples M = 400
        Filter update every N_s = 6 frames
        N_CG = 5
    """
    use_gpu = True
    device = 'cuda' if use_gpu else 'cpu'

    # Patch sampling parameters
    max_image_sample_size = 250 ** 2  # Maximum image sample size
    min_image_sample_size = 200 ** 2  # Minimum image sample size
    search_area_scale = 4.5  # Scale relative to target size

    # Conjugate Gradient parameters
    CG_iter = 5  # The number of Conjugate Gradient iterations in each update after the first frame
    init_CG_iter = 100  # The total number of Conjugate Gradient iterations used in the first frame
    init_GN_iter = 10  # The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
    post_init_CG_iter = 0  # CG iterations to run after GN
    fletcher_reeves = False  # Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
    standard_alpha = True  # Use the standard formula for computing the step length in Conjugate Gradient
    CG_forgetting_rate = 75  # Forgetting rate of the last conjugate direction
    precond_data_param = 0.3  # Weight of the data term in the preconditioner
    precond_reg_param = 0.15  # Weight of the regularization term in the preconditioner
    precond_proj_param = 35  # Weight of the projection matrix part in the preconditioner

    # Training parameters
    sample_memory_size = 200  # Memory size
    train_skipping = 10  # How often to run training (every n-th frame)

    # Detection parameters
    scale_factors = 1.02 ** torch.arange(-2, 3).float()  # What scales to use for localization
    score_upsample_factor = 1  # How much Fourier upsampling to use

    # Factorized convolution parameters
    update_projection_matrix = True  # Whether the projection matrix should be optimized or not
    projection_reg = 5e-8  # Regularization parameter of the projection matrix

    # Interpolation parameters
    interpolation_bicubic_a = -0.75  # The parameter for the bicubic interpolation kernel

    def __init__(self, feature):
        self.features = MultiResolutionExtractor(feature)
