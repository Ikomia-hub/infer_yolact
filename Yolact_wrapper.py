from ikomia import core
import torch
import torch.backends.cudnn as torch_cudnn
from collections import defaultdict
from Yolact.yolact_git.yolact import Yolact
from Yolact.yolact_git.utils.functions import SavePath
from Yolact.yolact_git.utils.augmentations import FastBaseTransform, FastBaseTransformCPU
from Yolact.yolact_git.data import cfg, set_cfg, COLORS
from Yolact.yolact_git.layers.output_utils import postprocess


color_cache = defaultdict(lambda: {})


def forward(src_img, param, graphics_output):
    img_numpy = None
    use_cuda = False
    if param.device == "cuda":
        use_cuda = torch.cuda.device_count() >= 1
        if use_cuda == False:
            raise ValueError("No CUDA driver!")
    
    init_config(param)

    with torch.no_grad():
        if use_cuda:
            torch_cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net = Yolact()
        device=torch.device(param.device)
        net.load_weights(param.model_path, device)
        net.eval()

        if use_cuda:
            net = net.cuda()

        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False

        frame = None
        if use_cuda:
            frame = torch.from_numpy(src_img).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
        else:
            frame = torch.from_numpy(src_img).float()
            batch = FastBaseTransformCPU()(frame.unsqueeze(0))

        predictions = net(batch)
        img_numpy = manage_outputs(predictions, frame, param, graphics_output)

    return img_numpy


def init_config(param):
    cfg.mask_proto_debug = False
    model_path = SavePath.from_str(param.model_path)
    config_name = model_path.model_name + "_config"
    set_cfg(config_name)


# Most parts come from function prep_display from yolact_git/eval.py
# Without command line arguments
def manage_outputs(predictions, img, param, graphics_output):
    crop_mask = True
    display_masks = True
    display_text = True
    display_bboxes = True
    display_scores = True

    # Put values in range [0 - 1]
    h, w, _ = img.shape

    # Post-processing
    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    t = postprocess(predictions, w, h, visualize_lincomb=False, crop_masks=crop_mask, score_threshold=param.confidence)
    cfg.rescore_bbox = save

    # Copy
    idx = t[1].argsort(0, descending=True)[:param.top_k]
    if cfg.eval_mask_branch:
        # Masks are drawn on the GPU, so don't copy
        masks = t[3][idx]

    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    # Filter available detections
    num_dets_to_consider = min(param.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < param.confidence:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    class_color = False

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])

            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Cumulative mask

        # For each object, we create a mask with label values for display purpose
        # For overlapping objects, we take those with the better probability with a MAX operator
        # that's why we reverse the mask weights
        mask_or = masks[0]*num_dets_to_consider
        for j in range(1, num_dets_to_consider):
            mask_or = torch.max(mask_or, masks[j] * (num_dets_to_consider-j))

        # Get the numpy array of the mask
        mask_numpy = mask_or.byte().cpu().numpy()

    colorvec = [[0,0,0]]

    if num_dets_to_consider == 0:
        return mask_numpy, colorvec

    if display_text or display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            colorvec.append(list(color))
            score = scores[j]

            if display_bboxes:
                rect_prop = core.GraphicsRectProperty()
                rect_prop.pen_color = list(color)
                graphics_box = graphics_output.addRectangle(float(x1), float(y1), float(x2-x1), float(y2-y1), rect_prop)
                graphics_box.setCategory(cfg.dataset.class_names[classes[j]])

            if display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if display_scores else _class

                text_prop = core.GraphicsTextProperty()
                text_prop.bold = True
                graphics_output.addText(text_str, float(x1), float(y1), text_prop)

    return mask_numpy, colorvec


