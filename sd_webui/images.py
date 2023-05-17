from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image, PngImagePlugin

    from sd_webui.processing import StableDiffusionProcessing

    def save_image(
        image: Image.Image,
        path: str,
        basename: str,
        seed: int | None = None,
        prompt: str = "",
        extension: str = "png",
        info: str | PngImagePlugin.iTXt = "",
        short_filename: bool = False,
        no_prompt: bool = False,
        grid: bool = False,
        pnginfo_section_name: str = "parameters",
        p: StableDiffusionProcessing | None = None,
        existing_info: dict | None = None,
        forced_filename: str | None = None,
        suffix: str = "",
        save_to_dirs: bool = False,
    ) -> tuple[str, str | None]:
        """Save an image.

        Args:
            image (`PIL.Image`):
                The image to be saved.
            path (`str`):
                The directory to save the image. Note, the option `save_to_dirs` will make the image to be saved into a sub directory.
            basename (`str`):
                The base filename which will be applied to `filename pattern`.
            seed, prompt, short_filename,
            extension (`str`):
                Image file extension, default is `png`.
            pngsectionname (`str`):
                Specify the name of the section which `info` will be saved in.
            info (`str` or `PngImagePlugin.iTXt`):
                PNG info chunks.
            existing_info (`dict`):
                Additional PNG info. `existing_info == {pngsectionname: info, ...}`
            no_prompt:
                TODO I don't know its meaning.
            p (`StableDiffusionProcessing`)
            forced_filename (`str`):
                If specified, `basename` and filename pattern will be ignored.
            save_to_dirs (bool):
                If true, the image will be saved into a subdirectory of `path`.

        Returns: (fullfn, txt_fullfn)
            fullfn (`str`):
                The full path of the saved imaged.
            txt_fullfn (`str` or None):
                If a text file is saved for this image, this will be its full path. Otherwise None.
        """

else:
    from modules.images import save_image
