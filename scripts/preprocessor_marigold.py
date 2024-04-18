import torch
import numpy as np

from marigold.model.marigold_pipeline import MarigoldPipeline

# sd-webui-controlnet
from internal_controlnet.external_code import Preprocessor, PreprocessorParameter
from scripts.utils import resize_image_with_pad

# A1111
from modules import devices


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


class PreprocessorMarigold(Preprocessor):
    def __init__(self, device=None):
        super().__init__(name = "depth_marigold")
        self.tags = ["Depth"]
        self.slider_resolution = PreprocessorParameter(
            label="Resolution",
            minimum=128,
            maximum=2048,
            value=768,
            step=8,
            visible=True,
        )
        self.slider_1 = PreprocessorParameter(
            label="Steps",
            minimum=1,
            maximum=50,
            value=20,
            step=1,
            visible=True,
        )
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list
        self.model = None
        self.device = (
            devices.get_device_for("controlnet")
            if device is None
            else torch.device("cpu")
        )

    def load_model(self):
        if self.model is None:
            self.model = MarigoldPipeline.from_pretrained(
                pretrained_path="Bingxin/Marigold",
                enable_xformers=False,
                noise_scheduler_type="DDIMScheduler",
            )

        return self.model.to(device=self.device)

    def unload_model(self):
        self.model.to(device="cpu")

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)

        pipeline = self.load_model()

        with torch.no_grad():
            img = (
                numpy_to_pytorch(input_image).movedim(-1, 1).to(device=pipeline.device)
            )
            img = img * 2.0 - 1.0
            depth = pipeline(img, num_inference_steps=slider_1, show_pbar=False)
            depth = 0.5 - depth * 0.5
            depth = depth.movedim(1, -1)[0].cpu().numpy()
            depth = np.concatenate([depth, depth, depth], axis=2)  # Expand to RGB
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        self.unload_model()
        return remove_pad(depth_image)


Preprocessor.add_supported_preprocessor(PreprocessorMarigold())
