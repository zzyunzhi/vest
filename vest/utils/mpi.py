"""Utilities for working with Multiplane Images (MPIs).
A multiplane image is a set of RGB + alpha textures, positioned as fronto-
parallel planes at specific depths from a reference camera. It represents a
lightfield and can be used to render new views from nearby camera positions
by warping each texture according to its plane homography and combining the
results with an over operation. More detail at:
    https://people.eecs.berkeley.edu/~tinghuiz/projects/mpi/
In this code, an MPI is represented by a tensor of layer textures and a tensor
of depths:
    layers: [..., L, 4, H, W] -- L is the number of layers, last dimension is
          typically RGBA but it can be any number of channels as long as the
          last channel is alpha.
    depths: [..., L] -- distances of the planes from the reference camera.
Layers and depths are stored back-to-front, i.e. farthest layer ("layer 0")
comes first. Typically the depths are chosen so that the corresponding
disparities (inverse depths) form an arithmetic sequence.
"""

import torch
import vest.utils.geometry as geometry

def layer_visibility(alphas):
    """Compute visibility for each pixel in each layer.
    Visibility says how unoccluded each pixel is by the corresponding pixels in
    front of it (i.e. those pixels with the same (x,y) position in subsequent
    layers). The front layer has visibility 1 everywhere since nothing can occlude
    it. Each other layer has visibility equal to the product of (1 - alpha) for
    all the layers in front of it.
    Args:
    alphas: [..., L, 1, H, W] Alpha channels for L layers, back to front.
    Returns:
    [..., L, 1, H, W] visibilities.
    """
    pseudo_ones = torch.ones_like(alphas)
    pseudo_ones = pseudo_ones[..., 0:1, :, :, :]
    alphas = torch.flip(alphas, [-4])
    alphas = 1.0 - alphas + 1e-8  # 1e-8 from mpi_extrapolation 2019 repo
    pseudo_alphas = torch.cat([pseudo_ones, alphas],dim=-4)
    vis = torch.cumprod(pseudo_alphas, axis=-4)
    vis = torch.flip(vis[..., :-1, :, :, :], [-4])
    return vis

def layer_weights(alphas):
    """Compute contribution weights for each layer from a set of alpha channels.
    The weights w_i for each layer are determined from the layer alphas so that
    to composite the layers we simple multiply each by its weight and add them
    up. In other words, the weight says how much each layer contributes to the
    final composed image.
    For alpha-blending, the weight of a layer at a point is its visibility at that
    point times its alpha at that point, i.e:
        alpha_i * (1 - alpha_i+1) * (1 - alpha_i+2) * ... (1 - alpha_n-1)
    If the first (i.e. the back) layer has alpha=1 everywhere, then the output
    weights will sum to 1 at each point.
    Args:
        alphas: [..., L, 1, H, W] Alpha channels for L layers, back to front.
    Returns:
        [..., L, 1, H, W] The resulting layer weights.
    """
    return alphas * layer_visibility(alphas)


def compose_hard(colors, alphas):
    """WARNINGS: No gradient for alphas!
    Args:
        colors: [..., L, C, H, W]
        alphas: [..., L, 1, H, W]

    Returns:
        [..., C, H, W]

    """
    assert colors.shape[:-3] == alphas.shape[:-3], (colors.shape, alphas.shape)
    assert colors.shape[-2:] == alphas.shape[-2:]

    transmittance = layer_weights(alphas)
    hard_transmittance = torch.argmax(transmittance, dim=-4, keepdim=True)
    hard_transmittance = hard_transmittance.expand(*colors.shape[:-4], 1, *colors.shape[-3:])

    return torch.gather(colors, dim=-4, index=hard_transmittance).squeeze(-4)


def compose_linear(colors, alphas):
    assert colors.shape[:-3] == alphas.shape[:-3], (colors.shape, alphas.shape)
    assert colors.shape[-2:] == alphas.shape[-2:]
    assert alphas.shape[-3] == 1
    return (colors * alphas).sum(-4)


def compose_soft(colors, alphas, split=False):
    # WARININGS: split should only be used for visualization purpose
    # otherwise use make_harmonic instead
    assert colors.shape[:-3] == alphas.shape[:-3], (colors.shape, alphas.shape)
    assert colors.shape[-2:] == alphas.shape[-2:]

    out = colors * layer_weights(alphas)
    if split:
        return out
    return out.sum(-4)


def compose_prob(colors, alphas):
    # for each pixel, sample a depth plane according to the prob distribution defined by transmittance
    # then assign rgb from that depth plane to the pixel
    # same as compose_hard but use softened max instead of argmax

    transmittance = layer_weights(alphas)  # (..., d, 1, h, w)
    try:
        transmittance = transmittance.moveaxis(-4, -1)
    except:
        transmittance = transmittance[None].flatten(0, -5)  # (b', d, 1, h, w)
        transmittance = transmittance.permute(0, 2, 3, 4, 1)
    ind = torch.multinomial(transmittance.flatten(0, -2), num_samples=1)  # (b'', d) -> (b'',)
    ind = ind.view(*alphas.shape[:-4], 1, *alphas.shape[-3:])  # (..., 1, 1, h, w)
    ind = ind.expand(*colors.shape[:-4], 1, *colors.shape[-3:])  # (..., 1, c, h, w)
    return torch.gather(colors, dim=-4, index=ind).squeeze(-4)


def compose_back_to_front(images):
    """Compose a set of images (for example, RGBA), back to front.
    Args:
    images: [..., L, C+1, H, W] Set of L images, with alpha in the last channel.
    Returns:
    [..., C, H, W] Composed image.
    """
    weights = layer_weights(images[..., -1:, :, :])
    return torch.sum(images[..., :-1, :, :] * weights, dim=-4)


def compute_disparity(alphas, front_depth, back_depth):
    raise NotImplementedError
    front_disparity = 1.0 / front_depth
    back_disparity = 1.0 / back_depth
    _, n_plane, _, _, _ = alphas.shape
    disparity_layers = torch.linspace(back_disparity, front_disparity, n_plane, device=alphas.device)
    disparity_layers = disparity_layers[..., :, None, None, None]
    # disparity = (disparity_layers * mpi_features['transmittance_layers']).sum(-4)
    disparity = (disparity_layers * layer_weights(alphas)).sum(-4)

    return disparity


def make_harmonic(rgba):
    """

    Args:
        rgba:

    Returns:
        An **equivalent** MPI representation such that the alpha layers is harmonic.
        As a consequence, its transmittance is uniformly 1/n_plane.
    """
    # with refinement, warped_alpha_layers == harmonic_alpha_layers
    #  (this is the whole point about refinement: FlowNet now does not need to take in 4 channels to be a valid "ground truth")
    #  the goal is that flow_network_temp matches FlowNet(tgt=rgba_of_next_t, ref=rgba_t)
    # canonical mpi: alpha = harmonic, compose operation == .mean(-4)  # note: not sum!
    n_plane = rgba.shape[-4]
    refined_colors = compose_soft(colors=rgba[..., :-1, :, :], alphas=rgba[..., -1:, :, :], split=True)

    harmonic_alphas = torch.FloatTensor([1 / l for l in range(1, n_plane + 1)])[:, None, None, None]
    for _ in rgba.shape[:-3]:
        harmonic_alphas.unsqueeze(0)
    harmonic_alphas = harmonic_alphas.expand(*rgba.shape[:-3], 1, *rgba.shape[-2:]).to(rgba)

    # n_plane is the correction term such that things are not dimmed out
    rgba_h = torch.cat([refined_colors * n_plane, harmonic_alphas], dim=-3)

    return rgba_h


def disparity_from_layers(layers, depths):
    """Compute disparity map from a set of MPI layers.
    From reference view.
    Args:
    layers: [..., L, C+1, H, W] MPI layers, back to front.
    depths: [..., L] depths for each layer.
    Returns:
    [..., 1, H, W] Single-channel disparity map from reference viewpoint.
    """
    disparities = 1.0 / depths
    # Add height, width and channel axes to disparities, so it can broadcast.
    disparities = disparities[..., :, None, None, None]
    weights = layer_weights(layers[..., :, -1:, :, :])

    # Weighted sum of per-layer disparities:
    return torch.sum(disparities * weights, dim=-4)


def make_depths(front_depth, back_depth, num_planes):
    # raise NotImplementedError
    """Returns a list of MPI plane depths, back to front.
    The first element in the list will be back_depth, and last will be
    near-depth, and in between there will be num_planes intermediate
    depths, which are interpolated linearly in disparity.
    Args:
    front_depth: The depth of the front-most MPI plane.
    back_depth: The depth of the back-most MPI plane.
    num_planes: The total number of planes to create.
    Returns:
    [num_planes] A tensor of depths sorted in descending order (so furthest
    first). This order is useful for back to front compositing.
    """
    assert front_depth < back_depth

    front_disparity = 1.0 / front_depth
    back_disparity = 1.0 / back_depth
    disparities = torch.linspace(back_disparity, front_disparity, num_planes)
    return 1.0 / disparities


def mpi_pred_to_mpi(ref_image, mpi_pred, which_color_pred, num_layers):
    """
    WARNINGS: assume mpi_pred comes after tanh!
    Args:
        ref_image: (b, c, h, w)
        mpi_pred:
            "bg": (b, c+d*2-1, h, w), concat([alphas*2-1, blend_weights*2-1, bg_rgb])
            "bg_no_blend": (b, c+d-1, h, w), concat([alphas*2-1, bg_rgb])
            "single": (b, d+c, h, w), concat([alphas*2-1, rgb])

    Returns:
        rgba_layers: (b, d, c+1, h, w)

    """
    bs, c, h, w = ref_image.shape
    d = num_layers

    if which_color_pred == 'bg':
        # d = (mpi_pred.shape[1] + 1 - c) // 2
        # assert mpi_pred.shape == (bs, c + d * 2 - 1, h, w)

        alphas = (mpi_pred[:, :d-1, :, :] + 1.) / 2.

        blend_weights = (mpi_pred[:, d-1:d*2-1, :, :] + 1.) / 2.  # (bs, d-1, h, w)
        pseudo_ones = torch.ones((bs, 1, h, w)).to(alphas.device)
        alphas = torch.cat([pseudo_ones, alphas], dim=1)  # (bs, d, h, w)

        bg_rgb = mpi_pred[:, d*2-1:d*2-1+c, :, :]
        fg_rgb = ref_image

        alphas = alphas.unsqueeze(2)  # (bs, d, 1, h, w)
        blend_weights = blend_weights.unsqueeze(2)  # (bs, d, 1, h, w)
        fg_rgb = fg_rgb.unsqueeze(1)
        bg_rgb = bg_rgb.unsqueeze(1)
        rgbs = blend_weights * fg_rgb + (1 - blend_weights) * bg_rgb  # (bs, d, c, h, w)

        unused = mpi_pred[:, d*2-1+c:, :, :]

    elif which_color_pred == 'bg_no_blend':
        # d = mpi_pred.shape[1] + 1 - c
        # assert mpi_pred.shape == (bs, c+d-1, h, w)

        alphas = (mpi_pred[:, :d-1, :, :] + 1.) / 2.
        pseudo_ones = torch.ones((bs, 1, h, w)).to(alphas.device)
        alphas = torch.cat([pseudo_ones, alphas], dim=1)  # (bs, d, h, w)

        bg_rgb = mpi_pred[:, d-1:d-1+c, :, :]
        fg_rgb = ref_image

        alphas = alphas.unsqueeze(2)  # (bs, d, 1, h, w)
        blend_weights = layer_visibility(alphas)  # (bs, d, 1, h, w)
        fg_rgb = fg_rgb.unsqueeze(1)
        bg_rgb = bg_rgb.unsqueeze(1)
        rgbs = blend_weights * fg_rgb + (1 - blend_weights) * bg_rgb  # (bs, d, c, h, w)

        unused = mpi_pred[:, d-1+c:, :, :]

    elif which_color_pred == 'single':
        # d = mpi_pred .shape[1] - c
        alphas = (mpi_pred[:, :d-1, :, :] + 1.) / 2.
        rgb = mpi_pred[:, d-1:d-1+c, :, :]

        alphas = alphas.unsqueeze(2)  # (bs, d, 1, h, w)
        rgb = rgb.unsqueeze(1)  # (bs, 1, c, h, w)
        rgbs = rgb.expand(bs, d, c, h, w)

        unused = mpi_pred[:, d-1+c:, :, :]

    elif which_color_pred == 'none':
        alphas = (mpi_pred[:, :d-1, :, :] + 1) / 2.
        pseudo_ones = torch.ones((bs, 1, h, w)).to(alphas)
        alphas = torch.cat([pseudo_ones, alphas], dim=1)  # (bs, d, h, w)

        rgbs = ref_image[:, None, :, :, :].expand(bs, d, c, h, w)
        alphas = alphas[:, :, None, :, :]

        unused = mpi_pred[:, d-1:, :, :]

    else:
        raise NotImplementedError

    return rgbs, alphas, unused
    # rgba_layers = torch.cat([rgbs, alphas], dim=2)  # (bs, d, c+1, h, w)
    #
    # return rgba_layers, unused


def render_layers(layers,
                  depths,
                  pose,
                  intrinsics,
                  target_pose,
                  target_intrinsics,
                  height=None,
                  width=None,
                  clamp=True):
    """Render target layers from MPI representation.

    Args:
      layers: [..., L, H, W, C] MPI layers, back to front.
      depths: [..., L] MPI plane depths, back to front.
      pose: [..., 3, 4] reference camera pose.
      intrinsics: [..., 4] reference intrinsics.
      target_pose: [..., 3, 4] target camera pose.
      target_intrinsics: [..., 4] target intrinsics.
      height: height to render to in pixels (or None for input height).
      width: width to render to in pixels (or None for input width).
      clamp: whether to clamp image coordinates (see geometry.sample_image doc),
        i.e. extending the image beyond its size or not.

    Returns:
      [..., L, height, width, C] The layers warped to the target view by applying
      an appropriate homography to each one.
    """
    source_to_target_pose = geometry.mat34_product(
        target_pose, geometry.mat34_pose_inverse(pose))

    # Add a dimension to correspond to L in the poses and intrinsics.
    pose = pose[Ellipsis, None, :, :]  # [..., 1, 3, 4]
    target_pose = target_pose[Ellipsis, None, :, :]  # [..., 1, 3, 4]
    intrinsics = intrinsics[Ellipsis, None, :]  # [..., 1, 4]
    target_intrinsics = target_intrinsics[Ellipsis, None, :]  # [..., 1, 4]

    # Fronto-parallel plane equations at the given depths, in the reference
    # camera's frame.
    normals = torch.tensor([0.0, 0.0, 1.0], device=layers.device).view(1, 3)
    depths = -depths[Ellipsis, None]  # [..., L, 1]
    normals, depths = geometry.broadcast_to_match(normals, depths, ignore_axes=1)
    planes = torch.cat([normals, depths], dim=-1)  # [..., L, 4]

    homographies = geometry.inverse_homography(pose, intrinsics, target_pose,
                                               target_intrinsics,
                                               planes)  # [..., L, 3, 3]
    # Each of the resulting [..., L] homographies knows how to inverse-warp one
    # of the [..., (H,W), L] images into a new [... (H',W')] target images.
    target_layers = geometry.homography_warp(
        layers, homographies, height=height, width=width, clamp=clamp)

    # The next few lines implement back-face culling.
    #
    # We don't want to render content that is behind the camera. (If we did, we
    # might see upside-down images of the layers.) A typical graphics approach
    # would be to test each pixel of each layer against a near-plane and discard
    # those that are in front of it. Here we implement something cheaper:
    # back-face culling. If the target camera sees the "back" of a layer then we
    # set that layer's alpha to zero. This is simple and sufficient in practice
    # to avoid nasty artefacts.

    # Convert planes to target camera space. target_planes is [..., L, 4]
    target_planes = geometry.mat34_transform_planes(source_to_target_pose, planes)

    # Fourth coordinate of plane is negative distance in front of the camera.
    # target_visible is [..., L]
    # target_visible = tf.cast(target_planes[Ellipsis, -1] < 0.0, dtype=tf.float32)
    target_visible = (target_planes[Ellipsis, -1] < 0.0).float()
    # per_layer_alpha is [..., L, 1, 1, 1]
    per_layer_alpha = target_visible[Ellipsis, None, None, None]
    # Multiply alpha channel by per_layer_alpha:
    non_alpha_channels = target_layers[Ellipsis, :-1]
    alpha = target_layers[Ellipsis, -1:] * per_layer_alpha

    target_layers = torch.cat([non_alpha_channels, alpha], dim=-1)
    return target_layers


if __name__ == "__main__":
    bs, d, h, w = 3, 4, 5, 5
    c = 3

    def over_composite(rgbas):
        # from https://github.com/google/stereo-magnification/blob/f2041f80ed8c340173a6048375ba900201c1f1e7/geometry/projector.py#L168

        """Combines a list of RGBA images using the over operation.
        Combines RGBA images from back to front with the over operation.
        The alpha image of the first image is ignored and assumed to be 1.0.
        Args:
          rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
        Returns:
          Composited RGB image.
        """
        for i in range(len(rgbas)):
            rgb = rgbas[i][:, :, :, 0:3]
            alpha = rgbas[i][:, :, :, 3:]
            if i == 0:
                output = rgb
            else:
                rgb_by_alpha = rgb * alpha
                output = rgb_by_alpha + output * (1.0 - alpha)
        return output

    img_final = torch.ones((bs, d-1+c, h, w)).normal_()
    img_final = torch.tanh(img_final)

    alpha_split = img_final[:, :-c, :, :]  # (bs, n_plane-1, h, w)
    alpha_split = (alpha_split + 1) * 0.5
    img_split = img_final[:, -c:, :, :]
    pseudo_ones = torch.ones((bs, 1, h, w)).to(alpha_split.device)
    alpha_split = torch.cat([pseudo_ones, alpha_split], dim=1)  # (bs, d, h, w)
    alpha_split = alpha_split[:, :, None, :, :]  # (bs, d, 1, h, w)
    output_split = img_split[:, None, :, :, :]  # (bs, 1, c, h, w)

    # blend = layer_visibility(alpha_split)
    # special case: when n_plane = 1, res = input.unsqueeze(1)
    # res = input_split * blend + output_split * (1 - blend)
    weights = layer_weights(alpha_split)
    res = (output_split * weights).sum(1)

    rgbas = torch.cat([output_split.expand(bs, d, c, h, w), alpha_split], dim=2)
    rgbas = rgbas.permute((1, 0, 3, 4, 2))
    res_ref = over_composite(rgbas)
    res_ref = res_ref.permute((0, 3, 1, 2))  # move to channel major

    assert res.shape == res_ref.shape == (bs, c, h, w)

    assert torch.allclose(res, res_ref)
    print('blending test passed')


    def infer_mpi(ref_image, mpi_pred):
        # https://github.com/google/stereo-magnification/blob/f2041f80ed8c340173a6048375ba900201c1f1e7/stereomag/mpi.py#L182

        # Rescale blend_weights to (0, 1)
        blend_weights = (mpi_pred[:, :, :, :d] + 1.) / 2.
        # Rescale alphas to (0, 1)
        alphas = (mpi_pred[:, :, :, d:d * 2 - 1] + 1.) / 2.
        alphas = torch.cat([torch.ones((bs, h, w, 1)), alphas], dim=-1)

        bg_rgb = mpi_pred[:, :, :, -3:]
        fg_rgb = ref_image
        # Assemble into an MPI (rgba_layers)
        for i in range(d):
            curr_alpha = alphas[:, :, :, i].unsqueeze(-1)  # (b, h, w, 1)
            wts = blend_weights[:, :, :, i].unsqueeze(-1)  # (b, h, w, 1)
            curr_rgb = wts * fg_rgb + (1 - wts) * bg_rgb
            curr_rgba = torch.cat([curr_rgb, curr_alpha], dim=3)  # (bs, h, w, c+1)
            if i == 0:
                rgba_layers = curr_rgba
            else:
                rgba_layers = torch.cat([rgba_layers, curr_rgba], dim=3)
        rgba_layers = rgba_layers.reshape(bs, h, w, d, 4)

        return rgba_layers

    ref_image = torch.ones((bs, c, h, w)).normal_() * 2 - 1  # (-1, 1)
    mpi_pred = torch.ones((bs, c+d*2-1, h, w)).normal_()
    mpi_pred = torch.tanh(mpi_pred)
    res = mpi_pred_to_mpi(ref_image=ref_image, mpi_pred=mpi_pred, which_color_pred='bg')

    # tf code
    ref_image = ref_image.movedim(1, -1)
    mpi_pred = mpi_pred.movedim(1, -1)
    rgba_layers = infer_mpi(ref_image, mpi_pred)
    rgba_layers = rgba_layers.permute(0, 3, 4, 1, 2)  # (bs, d, c+1, h, w)
    res_ref = rgba_layers

    assert res.shape == res_ref.shape == (bs, d, c+1, h, w)

    assert torch.allclose(res_ref, res)
    print('mpi_embd test passed')

    res = compose_back_to_front(rgba_layers)

    res_ref = over_composite(rgba_layers.permute(1, 0, 3, 4, 2))
    res_ref = res_ref.movedim(-1, 1)  # move to channel major

    assert res.shape == res_ref.shape == (bs, c, h, w)

    assert torch.allclose(res, res_ref)
    print('blending test passed')
