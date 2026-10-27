from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def emit(message: dict[str, Any]) -> None:
    print("WORKBENCH_JSON " + json.dumps(message, ensure_ascii=False), flush=True)


class UNOBridge:
    def __init__(self, config: dict[str, Any], device: str) -> None:
        self.config = config
        self.device_name = device

        os.environ.update(
            {
                "AE": str(config["ae_checkpoint"]),
                "FLUX_DEV": str(config["flux_checkpoint"]),
                "LORA": str(config["lora_checkpoint"]),
                "T5": str(config["t5_path"]),
                "CLIP": str(config["clip_path"]),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        repo_path = Path(config["repo_path"]).expanduser().resolve()
        # This bridge lives beside the workbench plugin package also named
        # ``uno``. Remove that script directory so Python resolves the research
        # repository's top-level ``uno`` package instead of the adapter package.
        bridge_directory = Path(__file__).resolve().parent
        sys.path = [
            entry
            for entry in sys.path
            if not entry or Path(entry).expanduser().resolve() != bridge_directory
        ]
        sys.path.insert(0, str(repo_path))
        sys.modules.pop("uno", None)

        import torch
        import torchvision.transforms.functional as tvf
        from einops import rearrange
        from PIL import Image
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not torch.cuda.is_available():
            raise RuntimeError("UNO 需要可用的 CUDA 设备")
        # Local experimental modules imported by the UNO pipeline overwrite
        # CUDA_VISIBLE_DEVICES at module scope. Initialize CUDA first so the
        # worker's assigned logical cuda:0 remains stable, then restore the env.
        torch.cuda.init()
        from uno.flux.pipeline import UNOPipeline, preprocess_ref
        from uno.flux.sampling import get_noise, get_schedule, prepare_multi_ip, unpack
        if visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        self.torch = torch
        self.tvf = tvf
        self.rearrange = rearrange
        self.Image = Image
        self.preprocess_ref = preprocess_ref
        self.get_noise = get_noise
        self.get_schedule = get_schedule
        self.prepare_multi_ip = prepare_multi_ip
        self.unpack = unpack
        self.device = torch.device(device)
        self.offload = bool(config.get("cpu_offload", True))
        # The research loader prints very large missing-key lists. When stdout is
        # also the JSON protocol pipe those messages can fill the pipe mid-line
        # and deadlock initialization, so discard only the third-party chatter.
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            self.pipeline = UNOPipeline(
                model_type=str(config.get("model_type", "flux-dev")),
                device=self.device,
                offload=self.offload,
                only_lora=bool(config.get("only_lora", True)),
                lora_rank=int(config.get("lora_rank", 512)),
            )
        if self.offload:
            # The upstream constructor initially places both encoders on CUDA.
            # Release them until text conditioning is prepared to lower idle VRAM.
            self.pipeline.offload_model_to_cpu(self.pipeline.t5, self.pipeline.clip)

    def _denoise(
        self,
        *,
        img: Any,
        img_ids: Any,
        txt: Any,
        txt_ids: Any,
        vec: Any,
        timesteps: list[float],
        guidance: float,
        ref_img: Any,
        ref_img_ids: Any,
        actions: Any = None,
        actions_ids: Any = None,
    ) -> Any:
        torch = self.torch
        total = len(timesteps) - 1
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        for step, (current, previous) in enumerate(
            zip(timesteps[:-1], timesteps[1:]), start=1
        ):
            timestep = torch.full(
                (img.shape[0],), current, dtype=img.dtype, device=img.device
            )
            prediction = self.pipeline.model(
                img=img,
                img_ids=img_ids,
                ref_img=ref_img,
                ref_img_ids=ref_img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=timestep,
                guidance=guidance_vec,
            )
            img = img + (previous - current) * prediction
            emit({"type": "progress", "step": step, "total": total})
        return img

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        width = max(16, 16 * (int(request.get("width", 512)) // 16))
        height = max(16, 16 * (int(request.get("height", 512)) // 16))
        seed = int(request.get("seed", 42))
        steps = max(1, int(request.get("num_inference_steps", 25)))
        extra = request.get("extra") or {}
        guidance = float(extra.get("guidance_scale", self.config.get("guidance_scale", 4.0)))
        pe = str(extra.get("pe", self.config.get("pe", "d")))

        paths = [Path(value) for value in request.get("input_images", [])]
        long_side = int(
            self.config.get("single_ref_size", 512)
            if len(paths) <= 1
            else self.config.get("multi_ref_size", 320)
        )
        references = []
        for path in paths:
            with self.Image.open(path) as opened:
                references.append(self.preprocess_ref(opened.convert("RGB"), long_side))

        with torch.inference_mode():
            noise = self.get_noise(
                1,
                height,
                width,
                device=self.device,
                dtype=torch.bfloat16,
                seed=seed,
            )
            timesteps = self.get_schedule(
                steps,
                (width // 8) * (height // 8) // (16 * 16),
                shift=True,
            )

            if self.offload:
                self.pipeline.ae.encoder = self.pipeline.ae.encoder.to(self.device)
            encoded_references = [
                self.pipeline.ae.encode(
                    (self.tvf.to_tensor(image) * 2.0 - 1.0)
                    .unsqueeze(0)
                    .to(self.device, torch.float32)
                ).to(torch.bfloat16)
                for image in references
            ]

            if self.offload:
                self.pipeline.offload_model_to_cpu(self.pipeline.ae.encoder)
                self.pipeline.t5 = self.pipeline.t5.to(self.device)
                self.pipeline.clip = self.pipeline.clip.to(self.device)
            conditioning = self.prepare_multi_ip(
                t5=self.pipeline.t5,
                clip=self.pipeline.clip,
                img=noise,
                prompt=str(request.get("prompt", "")),
                ref_imgs=encoded_references,
                pe=pe,
            )

            if self.offload:
                self.pipeline.offload_model_to_cpu(self.pipeline.t5, self.pipeline.clip)
                self.pipeline.model = self.pipeline.model.to(self.device)
            latent = self._denoise(
                **conditioning,
                timesteps=timesteps,
                guidance=guidance,
            )

            if self.offload:
                self.pipeline.offload_model_to_cpu(self.pipeline.model)
                self.pipeline.ae.decoder = self.pipeline.ae.decoder.to(self.device)
            latent = self.unpack(latent.float(), height, width)
            decoded = self.pipeline.ae.decode(latent)
            if self.offload:
                self.pipeline.offload_model_to_cpu(self.pipeline.ae.decoder)

            pixels = decoded.clamp(-1, 1)
            pixels = self.rearrange(pixels[-1], "c h w -> h w c")
            image = self.Image.fromarray((127.5 * (pixels + 1.0)).cpu().byte().numpy())
            output = Path(request["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output, format="PNG")

        return {
            "output": str(output),
            "metadata": {
                "model": "UNO",
                "task_type": request.get("task_type"),
                "reference_count": len(references),
                "seed": seed,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "cpu_offload": self.offload,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="uno")
    parser.parse_args()
    bridge: UNOBridge | None = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("type") == "initialize":
                bridge = UNOBridge(dict(message["config"]), str(message["device"]))
                emit({"type": "ready", "ok": True})
            elif message.get("type") == "generate":
                if bridge is None:
                    raise RuntimeError("UNO 尚未初始化")
                emit({"type": "result", **bridge.generate(dict(message["request"]))})
        except Exception as error:
            details = traceback.format_exc()
            print(details, file=sys.stderr, flush=True)
            emit(
                {
                    "type": "error",
                    "error": f"{type(error).__name__}: {error}\n{details[-4000:]}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

