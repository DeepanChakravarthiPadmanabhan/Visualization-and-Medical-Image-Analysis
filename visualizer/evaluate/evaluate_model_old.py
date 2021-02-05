import os
import logging
import gin
import typing

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from visualizer.loss_function.dice_loss import DiceLoss
from visualizer.loss_function.dice_loss import get_dice_coefficient

LOGGER = logging.getLogger(__name__)


@gin.configurable
def evaluate_model_old(
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

    if visualize and (
        not os.path.exists(os.path.join(image_output_path, "output_images"))
    ):
        os.makedirs(os.path.join(image_output_path, "output_images"))
        LOGGER.info(
            "Saving images in the directory: %s",
            os.path.join(image_output_path, "output_images"),
        )

    device = torch.device("cpu")
    state_test = torch.load(model_path, map_location=device)
    net.load_state_dict(state_test)
    net.eval()
    criterion = criterion()

    # instantiate dataset
    train_loader, valid_loader, test_loader = data_loaders

    LOGGER.info(
        "Evaluating Visualizer on BraTS 2019 images for brain tumor segmentation using the model, %s",
        model_path,
    )
    LOGGER.info("Results will be written to the path, %s", report_output_path)

    LOGGER.info("Ready to start evaluating!")

    # df_columns = ["loss", "precision", "recall", "f1-score"]
    # df_micro = pd.DataFrame(columns=df_columns)
    # df_macro = pd.DataFrame(columns=df_columns)
    df_dice = pd.DataFrame(columns=["loss", "dice"])
    # confusion_matrix_array = np.zeros((num_classes, num_classes))
    # precision_per_class = np.zeros((num_classes))
    # recall_per_class = np.zeros((num_classes))
    # f1score_per_class = np.zeros((num_classes))

    for data in test_loader:
        LOGGER.info("Predicting on image: %s", str(len(df_dice) + 1))

        images = data["image"]
        labels_segmentation = data["label"]
        labels_segmentation_argmax = torch.argmax(labels_segmentation, dim=1)
        outputs_segmentation = net(images)
        outputs_segmentation_argmax = outputs_segmentation.argmax(axis=1)

        if isinstance(criterion, DiceLoss):
            loss = criterion(outputs_segmentation, labels_segmentation)
        else:
            loss = criterion(outputs_segmentation, labels_segmentation_argmax)

        dice_value = get_dice_coefficient(outputs_segmentation, labels_segmentation)
        dice_value_mean = dice_value.mean()
        LOGGER.info(
            "Dice co-efficient for image %s: %s",
            str(len(df_dice) + 1),
            dice_value_mean.detach().numpy(),
        )
        LOGGER.info(
            "Dice loss for image %s: %s", str(len(df_dice) + 1), loss.detach().numpy()
        )
        df_dice.loc[len(df_dice)] = [
            loss.detach().numpy(),
            dice_value_mean.detach().numpy(),
        ]

        if visualize and len(df_dice) % 10 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(torch.squeeze(outputs_segmentation_argmax).detach().numpy())
            ax1.set_title("Prediction")
            ax2.imshow(torch.squeeze(labels_segmentation_argmax).detach().numpy())
            ax2.set_title("Target label")
            fig.suptitle("Visualizer result")
            LOGGER.info("Saving image number: %s", str(len(df_dice) + 1))
            fig.savefig(
                image_output_path + "/output_images/" + str(len(df_dice) + 1) + ".jpg"
            )
            plt.close(fig)

        outputs_seg_flatten = torch.flatten(outputs_segmentation_argmax, start_dim=1)
        labels_seg_flatten = torch.flatten(labels_segmentation_argmax, start_dim=1)

        # precision_micro, recall_micro, f1score_micro = calculate_metrics(
        #     labels_seg_flatten.detach().numpy(),
        #     outputs_seg_flatten.detach().numpy(),
        #     False,
        #     "micro",
        # )
        #
        # if precision_micro == recall_micro == f1score_micro == 0:
        #     continue
        #
        # precision_macro, recall_macro, f1score_macro = calculate_metrics(
        #     labels_seg_flatten.detach().numpy(),
        #     outputs_seg_flatten.detach().numpy(),
        #     False,
        #     "macro",
        # )
        #
        # if precision_macro == recall_macro == f1score_macro == 0:
        #     continue
        #
        # df_micro.loc[len(df_micro)] = [
        #     loss.detach().numpy(),
        #     precision_micro,
        #     recall_micro,
        #     f1score_micro,
        # ]
        #
        # df_macro.loc[len(df_macro)] = [
        #     loss.detach().numpy(),
        #     precision_macro,
        #     recall_macro,
        #     f1score_macro,
        # ]

        # image_precision, image_recall, image_f1score = calculate_metrics(
        #     labels_seg_flatten.detach().numpy(),
        #     outputs_seg_flatten.detach().numpy(),
        #     True,
        # )
        # precision_per_class = precision_per_class + image_precision
        # recall_per_class = recall_per_class + image_recall
        # f1score_per_class = f1score_per_class + image_f1score
        #
        # confusion_matrix_array = confusion_matrix_array + get_confusion_matrix(
        #     labels_seg_flatten.detach().numpy(), outputs_seg_flatten.detach().numpy()
        # )

    df_dice.loc["mean"] = df_dice.mean()
    # df_micro.loc["mean"] = df_micro.mean()
    # df_macro.loc["mean"] = df_macro.mean()
    # df_normalized_confusion_matrix = pd.DataFrame(
    #     confusion_matrix_array / len(df_micro)
    # )
    # df_precision_per_class = pd.DataFrame(precision_per_class / len(df_micro))
    # df_recall_per_class = pd.DataFrame(recall_per_class / len(df_micro))
    # df_f1score_per_class = pd.DataFrame(f1score_per_class / len(df_micro))

    excel_writer = pd.ExcelWriter(
        os.path.join(report_output_path, "report.xlsx"), engine="xlsxwriter"
    )
    df_dice.to_excel(excel_writer, sheet_name="dice")
    # df_micro.to_excel(excel_writer, sheet_name="micro")
    # df_macro.to_excel(excel_writer, sheet_name="macro")
    # df_normalized_confusion_matrix.to_excel(
    #     excel_writer, sheet_name="normalized_confusion_matrix"
    # )
    # df_precision_per_class.to_excel(excel_writer, sheet_name="precision_per_class")
    # df_recall_per_class.to_excel(excel_writer, sheet_name="recall_per_class")
    # df_f1score_per_class.to_excel(excel_writer, sheet_name="f1score_per_class")
    excel_writer.save()
    LOGGER.info("Results were written to %s", report_output_path)


def calculate_metrics(
    target: np.ndarray,
    output: np.ndarray,
    include_bg_class: bool = False,
    average: str = None,
    num_classes: int = 4,
) -> typing.Tuple:
    """
    Calculates the evaluation metrics precision, recall and f-score for the average
    method passed using sklearn.metrics.precision_recall_fscore_support.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction
        include_bg_class: bool
        Flag to whether include background class in metric calculation
        average: string
        Average argument for sklearn.metrics.precision_recall_fscore_support

    Returns:
        precision, recall, fscore: tuple
        Evaluation metrics: precision, recall and fscore respectively

    """
    if include_bg_class:
        start_label = 0
    else:
        start_label = 1
    metrics = precision_recall_fscore_support(
        target[0],
        output[0],
        labels=list(range(start_label, num_classes)),
        average=average,
    )

    return metrics[0], metrics[1], metrics[2]


def get_confusion_matrix(
    target: np.ndarray,
    output: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Calculates the confusion matrix normalized for each image with regard to all the classes
    method passed using sklearn.metrics.confusion_matrix.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction

    Returns:
        confusion_matrix_: np.ndarray
        Confusion matrix for all the classes

    """

    confusion_matrix_ = confusion_matrix(
        target[0], output[0], labels=list(range(0, num_classes)), normalize="all"
    )
    return confusion_matrix_
