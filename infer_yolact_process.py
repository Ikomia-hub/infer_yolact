from ikomia import utils, core, dataprocess
import copy
import os
import infer_yolact.yolact_wrapper as yw
from infer_yolact.yolact_git.data import cfg
import requests

# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class InferYolactParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here        
        models_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        self.model_path = models_folder + "/yolact_im700_54_800000.pth"
        self.conf_thres = 0.15
        self.top_k = 15
        self.mask_alpha = 0.45
        self.cuda = "cuda"

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.conf_thres = float(params["conf_thres"])
        self.top_k = float(params["top_k"])
        self.mask_alpha = float(params["mask_alpha"])
        self.cuda = params["cuda"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
            "conf_thres": str(self.conf_thres),
            "top_k": str(self.top_k),
            "mask_alpha": str(self.mask_alpha),
            "cuda": self.cuda
            }
        return params


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class InferYolact(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        
        # Add input/output of the process here
        self.net = None
        self.class_names = []

        # Create parameters class
        if param is None:
            self.set_param_object(InferYolactParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        # Load class names
        model_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        with open(model_folder + "/Coco_names.txt") as f:
            for row in f:
                self.class_names.append(row[:-1])

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get input :
        img_input = self.get_input(0)
        src_img = img_input.get_image()
        h, w, _ = src_img.shape

        # Get parameters :
        param = self.get_param_object()

        # Init instance segmentation output
        instance_output = self.get_output(1)
        instance_output.init("Yolact", 0, w, h)

        # Inference
        if not os.path.exists(param.model_path):
            print("Downloading model, please wait...")
            model_url = utils.get_model_hub_url() + "/" + self.name + "/yolact_im700_54_800000.pth"
            response = requests.get(model_url, stream=True)
            with open(param.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)          
            # self.download(model_url, param.model_path)

        num_dets_to_consider, masks, scores, boxes, classes = yw.forward(
            src_img,
            param,
            instance_output
        )
        self.set_names(list(cfg.dataset.class_names))
        names = list(cfg.dataset.class_names)
        for j in range(num_dets_to_consider):
            x1, y1, x2, y2 = boxes[j, :]
            score = scores[j]
            idx = names.index(cfg.dataset.class_names[classes[j]])
            self.add_object(j, 0, idx, float(score), float(x1), float(y1), float(x2-x1), float(y2-y1),
                            masks[j].byte().cpu().numpy())

        # Step progress bar:
        self.emit_step_progress()

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class InferYolactFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolact"
        self.info.short_description = "A simple, fully convolutional model for real-time instance segmentation."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.3.1"
        self.info.icon_path = "icon/icon.png"
        self.info.authors = "Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee"
        self.info.article = "YOLACT++: Better Real-time Instance Segmentation"
        self.info.journal = "ICCV"
        self.info.year = 2019
        self.info.license = "MIT License"
        self.info.documentation_link = "https://arxiv.org/abs/1912.06218"
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolact"
        self.info.original_repository = "https://github.com/dbolya/yolact"
        self.info.keywords = "CNN,detection,instance,segmentation,semantic,resnet"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INSTANCE_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferYolact(self.info.name, param)
