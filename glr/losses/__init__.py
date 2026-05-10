from glr.losses.consistency import slot_consistency_loss
from glr.losses.masked_pred import masked_prediction_loss
from glr.losses.vicreg import vicreg_loss

__all__ = ["vicreg_loss", "masked_prediction_loss", "slot_consistency_loss"]
