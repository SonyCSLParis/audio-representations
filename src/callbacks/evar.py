import pandas as pd
import subprocess
import tempfile
from typing import Any

try:
    import wandb
    _WANDB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _WANDB_AVAILABLE = False

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only


class EVARCallback(ModelCheckpoint):
    def __init__(self, *args, **kwargs) -> None:
        super(EVARCallback, self).__init__(*args, **kwargs)

        # attributes
        self.tempfile = tempfile.NamedTemporaryFile(delete=False).name + ".csv"
        self.eval_subprocess = None
        self._last_global_step_evaluated = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint and run evaluation in a separate process at the end of the training epoch."""
        super(EVARCallback, self).on_train_epoch_end(trainer, pl_module)
        if (trainer.global_rank == 0
                and self._last_global_step_evaluated != self._last_global_step_saved
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0):
            self.eval_subprocess = subprocess.Popen(['/bin/bash',
                                                     './scripts/quick_eval.sh',
                                                     self.last_model_path,
                                                     self.tempfile])
            self._last_global_step_evaluated = self._last_global_step_saved

    @rank_zero_only
    def on_train_batch_start(
            self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super(EVARCallback, self).on_train_batch_start(trainer, pl_module, batch, batch_idx)
        return_code = self.eval_subprocess.poll() if self.eval_subprocess is not None else None
        if return_code is None:  # evaluation is not finished, let's try again later
            return

        try:
            df = pd.read_csv(self.tempfile)

            # Calculate the average score
            average_score = df['score'].mean()

            # Create a new DataFrame for the average row
            average_row = pd.DataFrame({'task': ['average'], 'score': [average_score]})

            # Concatenate the original DataFrame with the new average row
            df = pd.concat([df, average_row], ignore_index=True)

            accuracies = dict(zip(df["task"], df["score"]))
            print("logging", accuracies)
            pl_module.log_dict({"accuracy/" + k: v for k, v in accuracies.items()})
        except Exception as e:
            print(f"Couldn't open tmpfile `{self.tempfile}`.")
            print(e)
        finally:
            self.eval_subprocess = None

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_end(trainer, pl_module)
        if trainer.global_rank != 0:
            return
        
        self.eval_subprocess = subprocess.Popen(['/bin/bash',
                                                './scripts/all_eval.sh',
                                                self.last_model_path,
                                                self.tempfile])
        self._last_global_step_evaluated = self._last_global_step_saved
    
        self.eval_subprocess.wait()
        print("process finished")

        try:
            df = pd.read_csv(self.tempfile).groupby("task", as_index=False)["score"].mean()

            # Calculate the average score
            average_score = df['score'].mean()

            # Create a new DataFrame for the average row
            average_row = pd.DataFrame({'task': ['average'], 'score': [average_score]})

            # Concatenate the original DataFrame with the new average row
            df = pd.concat([df, average_row], ignore_index=True)

            accuracies = dict(zip(df["task"], df["score"]))
            print("logging", accuracies)
            if _WANDB_AVAILABLE:
                table = wandb.Table(dataframe=df)
                wandb.log({
                    "downstream_acc": wandb.plot.bar(table, "task", "score", title="Downstream accuracy")
                })
            else:
                pl_module.logger.experiment.log_dict({"accuracy/" + k: v for k, v in accuracies.items()})
        except Exception as e:
            print(f"Couldn't open tmpfile `{self.tempfile}`.")
            print(e)
        finally:
            self.eval_subprocess = None
