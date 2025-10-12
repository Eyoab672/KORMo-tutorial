import os, shutil, tempfile, logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from concurrent import futures
from pathlib import Path
from typing import Sequence

from transformers import TrainerCallback, PreTrainedTokenizerBase, PreTrainedModel
from huggingface_hub import HfApi, upload_folder, upload_file
from huggingface_hub.errors import HfHubHTTPError
import huggingface_hub.utils as hf_hub_utils
import json
hf_hub_utils.disable_progress_bars()


class PushToHubCallback(TrainerCallback):
    def __init__(
        self,
        repo_id: str,
        hub_token: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        private: bool = False,
        gated = "manual",
        code_paths: Sequence[str] = None,
        **upload_kwargs,
    ):
        self.repo_id = repo_id
        self.hub_token = hub_token
        self.private = private
        self.gated = gated
        self.model = model
        self.tokenizer = tokenizer
        self.code_paths = [Path(p) for p in (code_paths or [])]
        self.upload_kwargs = upload_kwargs

        self.api = HfApi(token=hub_token)
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        self._latest_future = None

    def _ensure_repo(self):
        try:
            self.api.repo_info(self.repo_id, repo_type="model", token=self.hub_token)
            logging.info(f"Hub repo '{self.repo_id}' already exists.")
        except HfHubHTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                logging.info(f"Creating repo '{self.repo_id}' (private={self.private})")
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=self.private,
                    token=self.hub_token,
                    exist_ok=True,
                )
            else:
                raise
        self.api.update_repo_settings(
            repo_id=self.repo_id,
            repo_type="model",
            private=False,
            gated=self.gated,
            token=self.hub_token,
        )
    def _ensure_branch(self, revision: str):
        try:
            self.api.create_branch(
                repo_id=self.repo_id,
                branch=revision,
                repo_type="model",
                token=self.hub_token,
            )
            logging.info(f"Creating branch '{revision}'")
        except HfHubHTTPError as err:
            if err.response is None or err.response.status_code != 409:
                raise

    def _log_future(self, fut: futures.Future):
        if exc := fut.exception():
            logging.error("Async push failed:", exc_info=exc)

    def _register_model_to_autoclass(self):
        from kormo.model._modeling_kormo import KORMoForCausalLM, KORMoModel
        from kormo.model._configuration_kormo import KORMoConfig
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        AutoConfig.register("kormo", KORMoConfig)
        AutoModel.register(KORMoConfig, KORMoModel)
        AutoModelForCausalLM.register(KORMoConfig, KORMoForCausalLM)

        KORMoConfig.register_for_auto_class()
        KORMoModel.register_for_auto_class("AutoModel")
        KORMoForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    def _first_push_to_hub(self):
        self._register_model_to_autoclass()
        self.model.push_to_hub(
            repo_id=self.repo_id,
            token=self.hub_token,
            commit_message="Upload initial architecture and base weights",
        )
        self.tokenizer.push_to_hub(
            repo_id=self.repo_id,
            token=self.hub_token,
            commit_message="Add tokenizer files",
        )

    def on_init_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self._ensure_repo()
        future = self.executor.submit(self._first_push_to_hub)
        future.add_done_callback(self._log_future)
        self._latest_future = future
        logging.info(f"Pushed model code & tokenizer & weights (revision: Main)")


    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        ckpt_dir = Path(kwargs.get("checkpoint_dir", args.output_dir))
        if ckpt_dir.name != f"checkpoint-{state.global_step}":
            ckpt_dir = ckpt_dir / f"checkpoint-{state.global_step}"

        revision = f"step-{state.global_step}"
        commit_msg = f"Add checkpoint @ step {state.global_step}"
        self._ensure_branch(revision)

        future = self.executor.submit(
            upload_folder,
            repo_id=self.repo_id,
            folder_path=str(ckpt_dir),
            path_in_repo="",
            commit_message=commit_msg,
            revision=revision,
            token=self.hub_token,
            run_as_future=False,
            ignore_patterns=["optimizer.pt", "rng_state_*.pth"],
            **self.upload_kwargs,
        )
        future.add_done_callback(self._log_future)
        self._latest_future = future
        logging.info(f"Pushed checkpoint only (revision: {revision})")

    def on_train_end(self, args, state, control, **kwargs):
        if self._latest_future is not None:
            logging.info("Waiting for last async push…")
            self._latest_future.result()

class SaveDataStateCallback(TrainerCallback):
    def __init__(self, dataset, out_dir="ckpts"):
        self.ds, self.out = dataset, Path(out_dir)

    def on_save(self, args, state, control, **kwargs):
        import torch.distributed as dist, json, os
        ckpt_dir = Path(kwargs.get("checkpoint_dir", args.output_dir))
        if ckpt_dir.name != f"checkpoint-{state.global_step}":
            ckpt_dir = ckpt_dir / f"checkpoint-{state.global_step}"

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1

        local_state = self.ds.last_state
        gathered = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(gathered, local_state)
        else:
            gathered[0] = local_state

        if rank == 0:
            with open(ckpt_dir / "data_state.json", "w") as f:
                json.dump(gathered, f)