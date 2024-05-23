"""Test file."""

from main import image_size, num_channels, save_path, steps, variables
from modules.visualisation import Visualisation

# Visualise the result
vis_class = Visualisation(
    save_path,
    variables,
    (8, num_channels, image_size, image_size),
    steps,
)

if __name__ == "__main__":
    img = vis_class.get_image()  # generate image

    vis_class.tensor_to_image(
        img,
        "diffusion-model/assets/result.png",
    )  # save generated image
    vis_class.save_denoising_process(
        "diffusion-model/assets/denoising_process.gif",
    )  # save denoising process animation
