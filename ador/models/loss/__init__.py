from utils_.registry import Registry

LOSSES = Registry("losses")

from .AdMSLoss import AdMSoftmaxLoss
from .bmn_losses import TEMLoss, MidClsLoss, FinalClsLoss, PEMRegLoss, BoxRegLoss
from .tcn_losses import TCNL1Loss, TCNMSELoss, TCNBCELoss
from .temporal_hard_pair import TemporalHardPairLoss
