import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from visualizer.utils.constants import COLOR_MAPPING
from skimage import color


def sketch_gt_overlay(
    img, label, save_path="", modality="FLAIR", save_flag=False, show=False
):

    alpha = 0.6
    rows, cols = img.shape
    img_color = np.dstack((img, img, img))

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[label == 0] = COLOR_MAPPING["background"]  # Black block
    color_mask[label == 1] = COLOR_MAPPING["C1"]  # Red block
    color_mask[label == 2] = COLOR_MAPPING["C2"]  # Green block
    color_mask[label == 3] = COLOR_MAPPING["C3"]  # Blue block

    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    # Replace the hue and saturation of the original image with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    if save_flag:
        # Display the output
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"xticks": [], "yticks": []})
        ax1.set_aspect(1)
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title("Image (" + modality + ")")

        ax2.set_aspect(1)
        ax2.imshow(img_masked)
        ax2.set_title("Groundtruth mask")
        black_patch = mpatches.Patch(color="black", label="BG")
        red_patch = mpatches.Patch(color="red", label="NET")
        green_patch = mpatches.Patch(color="green", label="ED")
        blue_patch = mpatches.Patch(color="blue", label="ET")
        ax2.legend(
            handles=[black_patch, red_patch, green_patch, blue_patch],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()

    else:
        return img_masked


def post_process_gradient(gradient):
    """
    Exports the original gradient image
    :param gradient:
        Numpy array of the gradient with shape (4, 240, 240)
    :param filename:
    :return:
    """
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    return gradient


def get_positive_negative_saliency(gradient):
    """
    Generates positive and negative saliency maps based on the gradient
    :param gradient: numpy array
        Gradient of the operation to visualize
    :return:
    """
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return pos_saliency, neg_saliency
