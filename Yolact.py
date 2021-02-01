from ikomia import dataprocess
import Yolact_process as processMod
import Yolact_widget as widgetMod
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class Yolact(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.YolactProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.YolactWidgetFactory()
