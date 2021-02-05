import os
import logging
import gin
import typing

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from copy import deepcopy
import matplotlib.patches as mpatches

from torch.utils.data import Dataset

from visualizer.loss_function.dice_loss import get_dice_coefficient

from visualizer.utils.constants import MODALITY, COLOR_MAPPING

LOGGER = logging.getLogger(__name__)


@gin.configurable
def evaluate_model(
    model_path: str,
    report_output_path: str,
    image_output_path: str,
    criterion: torch.nn,
    net: torch.nn.Module,
    data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
    num_classes: int = 4,
    visualize: bool = False,
) -> None:
    """
    This function evaluates the model trained on a set of test image and provides a report with evaluation metrics.
    The evaluation metrics used are: Precision, Recall and F-score.
    The module also aids in visualizing the predictions and groundtruth labels.

    Args:
        model_path: string
        Path of the model to be used for inference
        report_output_path: string
        Path for writing the inference output report with evaluation metrics and visualization images
        criterion: torch.nn
        Loss type for evaluation
        net: torch.nn.Module
        Network architecture of the model trained
        visualize: bool
        To visualize the model predictions alongside groundtruth prediction
    """
    if not os.path.exists(os.path.dirname(report_output_path)):
        LOGGER.info(
            "Output directory does not exist. Creating directory %s",
            os.path.dirname(report_output_path),
        )
        os.makedirs(os.path.dirname(report_output_path))

    if visualize and (not os.path.exists(os.path.join(image_output_path))):
        os.makedirs(os.path.join(image_output_path))
        LOGGER.info(
            "Saving images in the directory: %s",
            os.path.join(image_output_path),
        )

    device = torch.device("cpu")
    state_test = torch.load(model_path, map_location=device)
    net.load_state_dict(state_test)
    net.eval()
    criterion = criterion()
    class_weights = torch.tensor([1, 1, 1, 1])

    # instantiate dataset
    train_loader, valid_loader, test_loader = data_loaders

    LOGGER.info(
        "Evaluating Visualizer on BraTS 2019 images for brain tumor segmentation using the model, %s",
        model_path,
    )
    LOGGER.info("Results will be written to the path, %s", report_output_path)

    LOGGER.info("Ready to start evaluating!")

    df_dice_all = pd.DataFrame(
        columns=["id", "class _0", "class_1", "class_2", "class_3"]
    )
    df_dice_percase = pd.DataFrame(
        columns=["id", "class _0", "class_1", "class_2", "class_3"]
    )
    running_dice_0 = []
    running_dice_1 = []
    running_dice_2 = []
    running_dice_3 = []

    for data in test_loader:
        count = 0

        dice_per_case_0 = []
        dice_per_case_1 = []
        dice_per_case_2 = []
        dice_per_case_3 = []

        LOGGER.info("Predicting on patient id: %s", data["id"][0])

        for image, label in zip(data["image"], data["label"]):
            count += 1
            LOGGER.info(
                "Predicting on image number: %s", data["id"][0] + "_" + str(count)
            )
            images = image
            outputs_segmentation = net(images)
            outputs_segmentation = (outputs_segmentation > 0.5).float()
            outputs_argmax = torch.squeeze(
                torch.argmax(outputs_segmentation, dim=1)
            ).detach()

            label_coded = torch.nn.functional.one_hot(label.long(), num_classes=4)
            label_coded = label_coded.squeeze(0)
            label_coded = label_coded.permute(2, 0, 1).to(dtype=torch.float32)
            dice_value = (
                get_dice_coefficient(
                    outputs_segmentation,
                    label_coded,
                    weight=class_weights,
                    device=device,
                )
                .detach()
                .numpy()
            )
            dice_value_0 = dice_value[0][0]
            dice_value_1 = dice_value[0][1]
            dice_value_2 = dice_value[0][2]
            dice_value_3 = dice_value[0][3]

            dice_per_case_0.append(dice_value_0)
            dice_per_case_1.append(dice_value_1)
            dice_per_case_2.append(dice_value_2)
            dice_per_case_3.append(dice_value_3)

            LOGGER.info(
                "Statistics of groundtruth image %s of patient id %s: %s",
                count,
                data["id"],
                label.unique(return_counts=True),
            )
            LOGGER.info(
                "Statistics of prediction on image %s of patient id %s: %s",
                count,
                data["id"],
                outputs_argmax.unique(return_counts=True),
            )
            LOGGER.info(
                "Dice co-efficient: (%s, %s, %s, %s)",
                dice_value_0,
                dice_value_1,
                dice_value_2,
                dice_value_3,
            )

            if visualize:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                prediction_numpy_array = outputs_argmax.numpy()
                label_numpy_array = torch.squeeze(label).detach().numpy()
                segmented_image = np.ones((prediction_numpy_array.shape) + (3,))
                segmented_image[prediction_numpy_array == 0] = COLOR_MAPPING[
                    "background"
                ]
                segmented_image[prediction_numpy_array == 1] = COLOR_MAPPING["C1"]
                segmented_image[prediction_numpy_array == 2] = COLOR_MAPPING["C2"]
                segmented_image[prediction_numpy_array == 3] = COLOR_MAPPING["C3"]
                ax1.imshow(segmented_image.astype(int))
                ax1.set_aspect(1)
                ax1.axis("off")
                ax1.set_title("Prediction")
                segmented_image = np.ones((label_numpy_array.shape) + (3,))
                segmented_image[label_numpy_array == 0] = COLOR_MAPPING["background"]
                segmented_image[label_numpy_array == 1] = COLOR_MAPPING["C1"]
                segmented_image[label_numpy_array == 2] = COLOR_MAPPING["C2"]
                segmented_image[label_numpy_array == 3] = COLOR_MAPPING["C3"]
                ax2.imshow(segmented_image.astype(int))
                ax2.set_aspect(1)
                ax2.axis("off")
                ax2.set_title("Target label")
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
                plt.suptitle("Visualizer result", x=0.5, y=0.84)
                plt.tight_layout()
                LOGGER.info("Saving image number: %s", data["id"][0] + "_" + str(count))
                plt.savefig(
                    image_output_path + data["id"][0] + "_" + str(count) + ".pdf"
                )
                plt.close(fig)

            df_dice_all.loc[len(df_dice_all)] = [
                data["id"][0] + "_slice_" + str(count),
                dice_value_0,
                dice_value_1,
                dice_value_2,
                dice_value_3,
            ]

        per_case_dice_class_0 = sum(dice_per_case_0) / 155
        per_case_dice_class_1 = sum(dice_per_case_1) / 155
        per_case_dice_class_2 = sum(dice_per_case_2) / 155
        per_case_dice_class_3 = sum(dice_per_case_3) / 155

        LOGGER.info(
            "Dice co-efficient for background class: %s ", per_case_dice_class_0
        )
        LOGGER.info("Dice co-efficient for class 1: %s", per_case_dice_class_1)
        LOGGER.info("Dice co-efficient for class 2: %s", per_case_dice_class_2)
        LOGGER.info("Dice co-efficient for class 4: %s", per_case_dice_class_3)

        running_dice_0.append(per_case_dice_class_0)
        running_dice_1.append(per_case_dice_class_1)
        running_dice_2.append(per_case_dice_class_2)
        running_dice_3.append(per_case_dice_class_3)

        df_dice_percase.loc[len(df_dice_percase)] = [
            data["id"][0],
            per_case_dice_class_0,
            per_case_dice_class_1,
            per_case_dice_class_2,
            per_case_dice_class_3,
        ]

    mean_dice_class_0 = sum(running_dice_0) / len(running_dice_0)
    mean_dice_class_1 = sum(running_dice_1) / len(running_dice_1)
    mean_dice_class_2 = sum(running_dice_2) / len(running_dice_2)
    mean_dice_class_3 = sum(running_dice_3) / len(running_dice_3)

    LOGGER.info("\n ####### Evaluation completed: Final statistics ####### ")
    LOGGER.info("Dice co-efficient for background class: %s", mean_dice_class_0)
    LOGGER.info("Dice co-efficient for class 1: %s", mean_dice_class_1)
    LOGGER.info("Dice co-efficient for class 2: %s", mean_dice_class_2)
    LOGGER.info("Dice co-efficient for class 4: %s", mean_dice_class_3)

    df_dice_all.loc["mean"] = [
        "0",
        df_dice_all.iloc[:, 1].mean(),
        df_dice_all.iloc[:, 2].mean(),
        df_dice_all.iloc[:, 3].mean(),
        df_dice_all.iloc[:, 4].mean(),
    ]
    df_dice_percase.loc["mean"] = [
        "0",
        df_dice_percase.iloc[:, 1].mean(),
        df_dice_percase.iloc[:, 2].mean(),
        df_dice_percase.iloc[:, 3].mean(),
        df_dice_percase.iloc[:, 4].mean(),
    ]
    excel_writer = pd.ExcelWriter(
        os.path.join(report_output_path, "report.xlsx"), engine="xlsxwriter"
    )
    df_dice_all.to_excel(excel_writer, sheet_name="all")
    df_dice_percase.to_excel(excel_writer, sheet_name="per_case")
    excel_writer.save()
    LOGGER.info("Results were written to %s", report_output_path)
