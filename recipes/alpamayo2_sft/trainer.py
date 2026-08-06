"""Custom HF Trainer for Alpamayo 2 Super SFT: surfaces the future_traj/others
loss split to console + W&B, mirroring alpamayo1_5_sft.trainer.ReasoningVLA_Trainer.

Depends on Alpamayo2SuperModelOutput carrying `loss_future_traj`/`loss_others`
(added to alpamayo2_super.models.alpamayo2_super -- a small, additive,
non-breaking field addition mirroring Alpamayo-1.5's ReasoningVLAOutput; see
that repo's models/alpamayo2_super.py). Recomputing this split independently in
the Trainer is not possible: `fuse_traj_tokens` mutates `input_ids` on a local
copy inside `forward()` (replacing placeholder tokens with the actual encoded
trajectory ids), so the pre-forward `tokenized_data["input_ids"]` this Trainer
sees does not carry the ids the loss was actually computed against.
"""

import logging

from transformers import Trainer

logger = logging.getLogger(__name__)


class Alpamayo2SFTTrainer(Trainer):
    """Trainer that additionally logs the future_traj/others loss split."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if getattr(model, "training", False):
            self._log_split_losses(outputs)
        return (loss, outputs) if return_outputs else loss

    def _log_split_losses(self, outputs) -> None:
        """Buffer per-step future_traj/others losses, emitted by `log()` below."""
        lf = getattr(outputs, "loss_future_traj", None)
        lo = getattr(outputs, "loss_others", None)
        if lf is None or lo is None:
            return
        if not hasattr(self, "_subloss_f"):
            self._subloss_f, self._subloss_o = [], []
        self._subloss_f.append(float(lf))
        self._subloss_o.append(float(lo))

    def log(self, logs, *args, **kwargs):
        """Attach mean buffered split-losses to the regular train-loss log event."""
        is_train_loss = "loss" in logs and "eval_loss" not in logs
        if getattr(self, "_subloss_f", None) and is_train_loss:
            logs = {
                **logs,
                "loss_future_traj": sum(self._subloss_f) / len(self._subloss_f),
                "loss_others": sum(self._subloss_o) / len(self._subloss_o),
            }
            self._subloss_f.clear()
            self._subloss_o.clear()
        return super().log(logs, *args, **kwargs)
