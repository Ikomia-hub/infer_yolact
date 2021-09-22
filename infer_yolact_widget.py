from ikomia import utils, core, dataprocess
from ikomia.utils import qtconversion
from infer_yolact.infer_yolact_process import InferYolactParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class InferYolactWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYolactParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Confidence
        label_confidence = QLabel("Confidence")
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0, 1)
        self.spin_confidence.setSingleStep(0.05)
        self.spin_confidence.setDecimals(2)
        self.spin_confidence.setValue(self.parameters.confidence)

        # Predictions count
        label_pred_count = QLabel("Max predictions count")
        self.spin_pred_count = QSpinBox()
        self.spin_pred_count.setRange(1, 100)
        self.spin_pred_count.setSingleStep(1)
        self.spin_pred_count.setValue(self.parameters.top_k)

        # Mask transparency
        label_mask_alpha = QLabel("Mask transparency")
        self.spin_mask_alpha = QDoubleSpinBox()
        self.spin_mask_alpha.setRange(0, 1)
        self.spin_mask_alpha.setSingleStep(0.05)
        self.spin_mask_alpha.setDecimals(2)
        self.spin_mask_alpha.setValue(self.parameters.mask_alpha)

        # Device
        self.checkbox = QCheckBox("CUDA")
        if self.parameters.device == "cuda":
            self.checkbox.setChecked(True)
        else:
            self.checkbox.setChecked(False)

        # Fill layout
        self.grid_layout.addWidget(label_confidence, 0, 0, 1, 1)
        self.grid_layout.addWidget(self.spin_confidence, 0, 1, 1, 1)
        self.grid_layout.addWidget(label_pred_count, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.spin_pred_count, 1, 1, 1, 1)
        self.grid_layout.addWidget(label_mask_alpha, 2, 0, 1, 1)
        self.grid_layout.addWidget(self.spin_mask_alpha, 2, 1, 1, 1)
        self.grid_layout.addWidget(self.checkbox, 3, 0, 1, 2)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.confidence = self.spin_confidence.value()
        self.parameters.top_k = self.spin_pred_count.value()
        self.parameters.mask_alpha = self.spin_mask_alpha.value()
        if self.checkbox.isChecked():
            self.parameters.device = "cuda"
        else:
            self.parameters.device = "cpu"

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class InferYolactWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_yolact"

    def create(self, param):
        # Create widget object
        return InferYolactWidget(param, None)
