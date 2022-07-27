import seaborn as sns
import matplotlib.pyplot as plt
from vest.utils.visualization.view_syn import ViewSynthesisHelper
import torch.nn.functional as F
from PIL import Image
from tu.loggers.utils import collect_tensor
from vest.third_party.mine.visualizations.image_to_video import disparity_normalization_vis, img_tensor_to_np
from vest.losses.similarity import apply_crop


helper = ViewSynthesisHelper()


def dump_kde(visualizer, metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 2, 2))
    for ax, (k, v) in zip(axes, metrics.items()):
        sns.kdeplot(v, ax=ax)
        ax.set_xlabel(k)
    helper.dump_table(visualizer, layout=[[fig]], col_type='figure', table_name='metrics')


def process_render_results(render_results, do_resize, do_crop):
    new_render_results = {}
    for k, v in render_results.items():
        if do_crop is True:
            v = apply_crop(v)
            # v = apply_crop_no_pad(v)
        if do_resize is not None:
            v = F.interpolate(v, size=do_resize, mode='bilinear', align_corners=False)
        new_render_results[k] = v
    return new_render_results


def get_render_results(inputs, outputs):
    render_results = {
        'src_imgs': inputs['prev_images'][:, -1],
        'tgt_imgs_syn': outputs['trg_rgb_syn_t'],
        'tgt_disparity_syn': outputs['pred_disparity'],  # it's src disparity
    }
    if 'stereo_prev_images' in inputs:
        render_results['tgt_imgs'] = inputs['stereo_prev_images'][:, -1]
    else:
        render_results['tgt_imgs'] = inputs['query_image']
        assert 'scale_factor' in outputs, outputs.keys()
    return render_results


def get_layout_row(render_results, similarity_metrics, do_resize=None, do_crop=False):
    render_results = process_render_results(render_results, do_resize=do_resize, do_crop=do_crop)
    tgt_imgs_syn = render_results["tgt_imgs_syn"] * 0.5 + 0.5
    tgt_disparity_syn = render_results["tgt_disparity_syn"]
    tgt_disparity_syn = disparity_normalization_vis(tgt_disparity_syn)
    src_imgs = render_results['src_imgs'] * 0.5 + 0.5
    tgt_imgs = render_results['tgt_imgs'] * 0.5 + 0.5
    layout_row = [src_imgs, tgt_imgs, tgt_imgs_syn, (tgt_imgs_syn - tgt_imgs).abs()]
    for i in range(len(layout_row)):
        layout_row[i] = Image.fromarray(collect_tensor(layout_row[i]))
    layout_row.append(Image.fromarray(img_tensor_to_np(tgt_disparity_syn)))

    scores = {k: v(tgt_imgs_syn, tgt_imgs).mean().item() for k, v in similarity_metrics.items()}
    return layout_row, scores


def apply_crop_no_pad(x):
    splat_bdry_ignore = 0.05

    *_, loss_h, loss_w = x.shape

    x_min = int(round(loss_w * splat_bdry_ignore))
    x_max = loss_w - x_min
    y_min = int(round(loss_h * splat_bdry_ignore))
    y_max = loss_h - y_min
    return x[..., y_min:y_max, x_min:x_max]
