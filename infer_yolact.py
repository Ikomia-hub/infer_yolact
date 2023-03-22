from ikomia import dataprocess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        from infer_yolact.infer_yolact_process import InferYolactFactory
        # Instantiate process object
        return InferYolactFactory()

    def get_widget_factory(self):
        from infer_yolact.infer_yolact_widget import InferYolactWidgetFactory
        # Instantiate associated widget object
        return InferYolactWidgetFactory()
