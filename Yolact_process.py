from ikomia import core, dataprocess
import copy
# Your imports below
import os
import Yolact_wrapper as yw


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class YolactParam(core.CProtocolTaskParam):

    def __init__(self):
        core.CProtocolTaskParam.__init__(self)
        # Place default value initialization here        
        models_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        self.model_path = models_folder + "/yolact_im700_54_800000.pth"
        self.confidence = 0.15
        self.top_k = 15
        self.mask_alpha = 0.45
        self.device = "cuda"

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.confidence = float(paramMap["confidence"])
        self.top_k = float(paramMap["top_k"])
        self.mask_alpha = float(paramMap["mask_alpha"])
        self.device = paramMap["device"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = core.ParamMap()
        paramMap["confidence"] = str(self.confidence)
        paramMap["top_k"] = str(self.top_k)
        paramMap["mask_alpha"] = str(self.mask_alpha)
        paramMap["device"] = self.device
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class YolactProcess(dataprocess.CImageProcess2d):

    def __init__(self, name, param):
        dataprocess.CImageProcess2d.__init__(self, name)
        
        # Add input/output of the process here
        self.setOutputDataType(core.IODataType.IMAGE_LABEL, 0)
        self.addOutput(dataprocess.CImageProcessIO(core.IODataType.IMAGE))
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        self.net = None
        self.class_names = []

        # Create parameters class
        if param is None:
            self.setParam(YolactParam())
        else:
            self.setParam(copy.deepcopy(param))

        # Load class names
        model_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        with open(model_folder + "/Coco_names.txt") as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        img_input = self.getInput(0)
        src_img = img_input.getImage()

        # Get parameters :
        param = self.getParam()

        # Init graphics output
        graphics_output = self.getOutput(2)
        graphics_output.setNewLayer("Yolact")
        graphics_output.setImageIndex(1)

        # Inference
        mask, colorvec = yw.forward(src_img, param, graphics_output)

        # Step progress bar:
        self.emitStepProgress()

        # Get mask output :
        mask_output = self.getOutput(0)
        mask_output.setImage(mask)
        self.setOutputColorMap(1, 0, colorvec)

        # Get image output :
        self.forwardInputImage(0, 1)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class YolactProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Yolact"
        self.info.shortDescription = "A simple, fully convolutional model for real-time instance segmentation."
        self.info.description = "We present a simple, fully-convolutional model for real-time (>30 fps) instance " \
                                "segmentation that achieves competitive results on MS COCO evaluated on a single " \
                                "Titan Xp, which is significantly faster than any previous state-of-the-art approach. " \
                                "Moreover, we obtain this result after training on only one GPU. We accomplish this " \
                                "by breaking instance segmentation into two parallel subtasks: (1) generating a set " \
                                "of prototype masks and (2) predicting per-instance mask coefficients. Then we produce " \
                                "instance masks by linearly combining the prototypes with the mask coefficients. " \
                                "We find that because this process doesn't depend on repooling, this approach produces " \
                                "very high-quality masks and exhibits temporal stability for free. Furthermore, we " \
                                "analyze the emergent behavior of our prototypes and show they learn to localize " \
                                "instances on their own in a translation variant manner, despite being fully-convolutional. " \
                                "We also propose Fast NMS, a drop-in 12 ms faster replacement for standard NMS that " \
                                "only has a marginal performance penalty. Finally, by incorporating deformable " \
                                "convolutions into the backbone network, optimizing the prediction head with better " \
                                "anchor scales and aspect ratios, and adding a novel fast mask re-scoring branch, " \
                                "our YOLACT++ model can achieve 34.1 mAP on MS COCO at 33.5 fps, which is fairly " \
                                "close to the state-of-the-art approaches while still running at real-time."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.iconPath = "icon/icon.png"
        self.info.authors = "Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee"
        self.info.article = "YOLACT++: Better Real-time Instance Segmentation"
        self.info.journal = "ICCV"
        self.info.year = 2019
        self.info.license = "MIT License"
        self.info.documentationLink = "https://arxiv.org/abs/1912.06218"
        self.info.repository = "https://github.com/dbolya/yolact"
        self.info.keywords = "CNN,detection,instance,segmentation,semantic,resnet"

    def create(self, param=None):
        # Create process object
        return YolactProcess(self.info.name, param)
