from ikomia import dataprocess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class Yolact(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from Yolact.Yolact_process import YolactProcessFactory
        # Instantiate process object
        return YolactProcessFactory()

    def getWidgetFactory(self):
        from Yolact.Yolact_widget import YolactWidgetFactory
        # Instantiate associated widget object
        return YolactWidgetFactory()
