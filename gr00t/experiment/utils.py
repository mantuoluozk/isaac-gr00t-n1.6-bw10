from pathlib import Path
import logging
import shutil

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class BestMetricCheckpointCallback(TrainerCallback):
    """This callback saves the best checkpoint based on the metric."""

    def __init__(
        self, metric_name: str, greater_is_better: bool = True, exp_cfg_dir: Path | None = None
    ):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self.exp_cfg_dir = exp_cfg_dir
        self._best_checkpoint_dir = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        if state.is_world_process_zero and metrics is not None:
            current_metric = metrics.get(self.metric_name, None)
            if current_metric is not None:
                is_better = (
                    self.greater_is_better
                    if current_metric > self.best_metric
                    else not self.greater_is_better
                )
                if is_better:
                    self.best_metric = current_metric
                    best_checkpoint_dir = (
                        Path(args.output_dir)
                        / f"checkpoint-{state.global_step}-best-{self.metric_name}_{current_metric}"
                    )
                    best_checkpoint_dir.mkdir(exist_ok=True)
                    model.save_pretrained(best_checkpoint_dir)
                    # Copy experiment config directory if provided
                    if self.exp_cfg_dir is not None:
                        exp_cfg_dst = best_checkpoint_dir / self.exp_cfg_dir.name
                        if self.exp_cfg_dir.exists():
                            print(
                                f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                            )
                            shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

                    print(
                        f"Best checkpoint saved to {best_checkpoint_dir} with metric {self.metric_name} = {current_metric}"
                    )

                    if (
                        self._best_checkpoint_dir is not None
                        and Path(self._best_checkpoint_dir).exists()
                    ):
                        shutil.rmtree(self._best_checkpoint_dir)

                    self._best_checkpoint_dir = str(best_checkpoint_dir)


class CodeBackupCallback(TrainerCallback):
    def __init__(self, output_dir: Path | str, target_step: int = 500):
        self.output_dir = Path(output_dir)
        self.target_step = target_step
        self._backup_done = False

    def copy_gr00t_folder(self, output_dir: str | Path) -> Path | None:
        output_dir = Path(output_dir)
        gr00t_src_dir = Path(__file__).parent.parent
        
        gr00t_dst_base = output_dir / "gr00t_snapshot"
        gr00t_dst_zip = output_dir / "gr00t_snapshot.zip"
        
        try:
            # Remove existing zip file
            if gr00t_dst_zip.exists():
                gr00t_dst_zip.unlink()
            shutil.make_archive(
                str(gr00t_dst_base),  # base name without extension
                'zip',
                root_dir=gr00t_src_dir.parent,  # start from parent dir
                base_dir=gr00t_src_dir.name,  # include only the gr00t folder
            )
            return gr00t_dst_zip
        except Exception as e:
            return None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._backup_done:
            return

        if state.global_step == self.target_step and state.is_world_process_zero:
            result = self.copy_gr00t_folder(self.output_dir)
            if result is not None:
                print(f"[CodeBackupCallback] Successfully backed up gr00t to {result}")
            else:
                print("[CodeBackupCallback] Failed to backup gr00t source code")
            self._backup_done = True

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after the trainer saves a checkpoint. Copies gr00t_snapshot.zip to each checkpoint folder."""
        if not state.is_world_process_zero:
            return

        gr00t_snapshot_src = self.output_dir / "gr00t_snapshot.zip"
        
        if not gr00t_snapshot_src.exists():
            return

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        gr00t_snapshot_dst = checkpoint_dir / "gr00t_snapshot.zip"

        try:
            if checkpoint_dir.exists():
                shutil.copy2(gr00t_snapshot_src, gr00t_snapshot_dst)
                print(f"[CodeBackupCallback] Copied gr00t_snapshot.zip to {gr00t_snapshot_dst}")
        except Exception as e:
            logging.error(f"[CodeBackupCallback] Failed to copy gr00t_snapshot.zip to checkpoint: {e}")
