import torch
from torch.nn import functional as F

from .tensorlist import TensorList


def sample_patch(
        im: torch.Tensor,
        pos: torch.Tensor,
        sample_sz: torch.Tensor,
        output_sz: torch.Tensor = None):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
    """

    # copy and convert
    posl = pos.long().clone()

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df  # new size

    # Do downsampling
    if df > 1:
        os = posl % df  # offset
        posl = (posl - os) // df  # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]  # downsample
    else:
        im2 = im

    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl // 2 + 1

    # Get image patch
    im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]),
                     'replicate')

    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1, 4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample
    im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear', align_corners=False)

    return im_patch, patch_coord


class MultiResolutionExtractor:

    def __init__(self, feature):
        self.feature = feature

    def size(self, input_sz):
        return TensorList([input_sz]).unroll()

    def extract(self, im, pos, scales, image_sz):
        """Extract features.
        args:
            im: Image.
            pos: Center position for extraction.
            scales: Image scales to extract features from.
            image_sz: Size to resize the image samples to before extraction.
        """
        if isinstance(scales, (int, float)):
            scales = [scales]

        # Get image patches
        patch_iter, coord_iter = zip(*(sample_patch(im, pos, s * image_sz, image_sz) for s in scales))
        im_patches = torch.cat(list(patch_iter))
        patch_coords = torch.cat(list(coord_iter))

        # Compute features
        feature_map = TensorList([self.feature.get_feature(im_patches)]).unroll()

        return feature_map, patch_coords

    def extract_transformed(self, im, pos, scale, image_sz):
        """Extract features from a set of transformed image samples.
        args:
            im: Image.
            pos: Center position for extraction.
            scale: Image scale to extract features from.
            image_sz: Size to resize the image samples to before extraction.
        """

        # Get image patch
        im_patch, _ = sample_patch(im, pos, scale * image_sz, image_sz)

        # Compute features
        feature_map = TensorList([self.feature.get_feature(im_patch)]).unroll()

        return feature_map
