import os
import gin
import logging
import torch
from torch.nn import ReLU
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from visualizer.utils.constants import MODALITY, FLAG_TO_STUDY, TARGET_CLASS_TO_STUDY
from visualizer.utils.post_processing import (
    sketch_gt_overlay,
    post_process_gradient,
    get_positive_negative_saliency,
)
from copy import deepcopy
import cv2

import torch.nn as nn

LOGGER = logging.getLogger(__name__)

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.eval()

        self.update_relus()
        self.hook_layers()

    def get_all_conv_layers(self, model):
        """
        Parameters:
        -----------
        model: torch.nn
            PyTorch CNN model

        Return:
        -------
        name_of_layers: typing.List
            Name of the CNN layers
        """

        name_of_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                module, torch.nn.ConvTranspose2d
            ):
                name_of_layers.append(module)
        return name_of_layers

    def hook_layers(self):
        def gradient_hook_function(module, grad_in, grad_out):
            # Grad-in is the gradient with respect to the image pixels in the shape (1, 4, h, w)
            # Grad-out is the gradient with respect to the 1st conv layer
            self.gradients = grad_in[0]

        def logit_hook_function(module, input, output):
            self.logit_output = output.detach()

        conv_layers = self.get_all_conv_layers(self.model)
        first_layer = conv_layers[0]
        logit_layer = conv_layers[-1]
        first_layer.register_backward_hook(gradient_hook_function)
        logit_layer.register_forward_hook(logit_hook_function)

    def update_relus(self):
        """
        Updates relu activation by doing the following tasks:
        1. Stores output in forward pass
        2. Imputes zero for gradient values that are less than zero

        Returns:

        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            # Extract f_i^l
            corresponding_forward_output = self.forward_relu_outputs[-1]
            # Extract (f_i^l > 0) mask
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            # R_i^l = (f_i^l > 0) * (R_i^(l + 1) > 0)
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0
            )
            # Remove f_i^l
            del self.forward_relu_outputs[-1]
            # This return value will be multiplied by the gradients in the backward pass
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Stores results of forward pass
            :param module:
            :param ten_in:
            :param ten_out:
            :return:
            """
            self.forward_relu_outputs.append(ten_out)

        for pos, module in self.model.named_modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, image, predictions, target_class=1, flag_to_study=4):
        self.model.zero_grad()
        out = self.model(image)
        self.target_class = target_class
        self.flag_to_study = flag_to_study

        if flag_to_study == 1:
            LOGGER.info("Performing visualization study %s on class %s", flag_to_study, target_class)
            loss_final_channels = torch.zeros(self.logit_output.shape)
            loss_final_channels[0, target_class, ::] = self.logit_output[
                                                         0, target_class, ::
                                                         ]
            loss_final_channels_avg = torch.mean(
                loss_final_channels.view(loss_final_channels.size(0), loss_final_channels.size(1), -1), dim=2)
            loss_final = torch.FloatTensor(loss_final_channels_avg[0])
            print('Average logit: ', loss_final)
            loss_final.unsqueeze_(0)
            loss_final.unsqueeze_(2)
            loss_final.unsqueeze_(2)

        elif flag_to_study==2:
            LOGGER.info("Performing visualization study %s on class %s", flag_to_study, target_class)
            loss_final_channels = torch.zeros(self.logit_output.shape)
            loss_final = torch.FloatTensor([0, 0, 0, 0])
            mask = predictions[0, target_class, ::]
            idxs = torch.where(mask == 1)
            if len(idxs[0]) != 0:
                loss_final_channels[
                    0, target_class, idxs[0], idxs[1]
                ] = self.logit_output[0, target_class, idxs[0], idxs[1]]

                loss_final_channels_avg = torch.sum(loss_final_channels.view(loss_final_channels.size(0), loss_final_channels.size(1), -1), dim=2)
                loss_final_channels_avg = loss_final_channels_avg / len(idxs[0])
                loss_final = torch.FloatTensor(loss_final_channels_avg[0])

            print('Average logit: ', loss_final)
            loss_final.unsqueeze_(0)
            loss_final.unsqueeze_(2)
            loss_final.unsqueeze_(2)


        elif flag_to_study == 3:
            # Research question 3
            LOGGER.info("Performing visualization study %s on class %s", flag_to_study, target_class)
            logit_image = self.logit_output[0, target_class, ::]
            logit_image = logit_image.detach().numpy()

            # Select ROI
            coordinates = cv2.selectROI("Image", logit_image)
            LOGGER.info("User selected region (x, y, w, h): %s", coordinates)

            # Crop image
            # coordinates -> ( x, y, w, h )
            # crop image -> (y: y + h, x: x + w)
            # crop_image = logit_image[int(coordinates[1]):int(coordinates[1] + coordinates[3]),
            #              int(coordinates[0]):int(coordinates[0] + coordinates[2])]
            # Display cropped image
            # cv2.imshow("Image", crop_image)
            # cv2.waitKey(0)

            loss_final_channels = torch.zeros(self.logit_output.shape)
            loss_final_channels[
                0,
                target_class,
                int(coordinates[1]) : int(coordinates[1] + coordinates[3]),
                int(coordinates[0]) : int(coordinates[0] + coordinates[2]),
            ] = self.logit_output[
                0,
                target_class,
                int(coordinates[1]) : int(coordinates[1] + coordinates[3]),
                int(coordinates[0]) : int(coordinates[0] + coordinates[2]),
            ]
            loss_final_channels_avg = torch.mean(
                loss_final_channels.view(loss_final_channels.size(0), loss_final_channels.size(1), -1), dim=2)
            loss_final = torch.FloatTensor(loss_final_channels_avg[0])
            print('Average logit: ', loss_final)
            loss_final.unsqueeze_(0)
            loss_final.unsqueeze_(2)
            loss_final.unsqueeze_(2)

        else:
            raise NotImplementedError("Only 3 visualization options available")

        out.backward(loss_final)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


@gin.configurable
def guided_backpropagate(
    model_path,
    dataloader,
    net,
    report_output_path,
    image_output_path,
    visualize,
    target_class_to_study=1,
    flag_to_study=1,
    idx=150,
    grad_visualizer_modality=1,
    show=False,
    visualize_saliency=False,
):

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

    test_loader = dataloader[2]
    device = torch.device("cpu")

    for data in test_loader:
        count = 0
        state_test = torch.load(model_path, map_location=device)
        net.load_state_dict(state_test)

        for image, label in zip(data["image"][idx], data["label"][idx]):
            count += 1

            image_grad = deepcopy(image)
            new_net = deepcopy(net)
            new_net.mask_layer._modules['7'] = nn.AvgPool2d(240)

            image.unsqueeze_(0)
            model_output = net(image)
            predictions_raw = (model_output > 0.5).float()
            mask_0 = torch.where(predictions_raw[0, 0] == 1)
            mask_1 = torch.where(predictions_raw[0, 1] == 1)
            mask_2 = torch.where(predictions_raw[0, 2] == 1)
            mask_3 = torch.where(predictions_raw[0, 3] == 1)
            print('Label counts for prediction in GBP: ', len(mask_0[0]), len(mask_1[0]), len(mask_2[0]), len(mask_3[0]))
            predictions = torch.argmax(predictions_raw[0], dim=0)
            print('Label counts in final prediction labels: ', torch.unique(predictions, return_counts=True))

            GBP = GuidedBackprop(new_net)

            image_grad.requires_grad_(True)
            image_grad = image_grad.unsqueeze(0)
            grad = GBP.generate_gradients(
                image_grad, predictions=predictions_raw, target_class=target_class_to_study, flag_to_study=flag_to_study
            )

            processed_grad = []
            for i in range(4):
                processed_grad.append(post_process_gradient(grad[i]))

            if visualize:

                fig, ax = plt.subplots(2, 3)

                img_masked = sketch_gt_overlay(
                    image[0][grad_visualizer_modality].detach().numpy(),
                    label,
                    MODALITY[grad_visualizer_modality],
                )

                predicted_mask = sketch_gt_overlay(
                    image[0][grad_visualizer_modality].detach().numpy(),
                    predictions.detach().numpy(),
                    MODALITY[grad_visualizer_modality],
                )

                ax[0,0].set_aspect(1)
                ax[0,0].imshow(processed_grad[0], cmap='gray')
                ax[0,0].axis("off")
                ax[0,0].set_title("FLAIR")

                ax[0,1].set_aspect(1)
                ax[0,1].imshow(processed_grad[1], cmap='gray')
                ax[0,1].axis("off")
                ax[0,1].set_title("T1")

                ax[1,0].set_aspect(1)
                ax[1,0].imshow(processed_grad[2], cmap='gray')
                ax[1,0].axis("off")
                ax[1,0].set_title("T1-CE")

                ax[1,1].set_aspect(1)
                ax[1,1].imshow(processed_grad[3], cmap='gray')
                ax[1,1].axis("off")
                ax[1,1].set_title("T2")

                ax[0, 2].set_aspect(1)
                ax[0, 2].imshow(img_masked)
                ax[0, 2].axis("off")
                ax[0, 2].set_title(
                    "Image (" + MODALITY[grad_visualizer_modality] + ") + Mask"
                )
                black_patch = mpatches.Patch(color="black", label="BG")
                red_patch = mpatches.Patch(color="red", label="NET")
                green_patch = mpatches.Patch(color="green", label="ED")
                blue_patch = mpatches.Patch(color="blue", label="ET")
                ax[0, 2].legend(
                    handles=[black_patch, red_patch, green_patch, blue_patch],
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                )

                ax[1, 2].set_aspect(1)
                ax[1, 2].imshow(predicted_mask)
                ax[1, 2].axis("off")
                ax[1, 2].set_title("Predicted mask")

                plt.suptitle(
                    "Gradients: "
                    + FLAG_TO_STUDY[flag_to_study]
                    + ", Class: "
                    + TARGET_CLASS_TO_STUDY[target_class_to_study],
                    x=0.5,
                    y=0.95,
                )
                plt.tight_layout()
                plt.savefig(
                    image_output_path
                    + data["id"][0]
                    + "_mask_grad_flag_"
                    + FLAG_TO_STUDY[flag_to_study]
                    + "_class_"
                    + TARGET_CLASS_TO_STUDY[target_class_to_study]
                    + "_count_"
                    + str(count)
                    + ".pdf",
                    dpi=300,
                )
                if show:
                    plt.show()
                plt.close(fig)

            if visualize_saliency:

                pos_grad = []
                neg_grad = []
                for i in range(4):
                    pos_saliency, neg_saliency = get_positive_negative_saliency(
                        grad[i]
                    )
                    pos_grad.append(pos_saliency)
                    neg_grad.append(neg_saliency)

                fig, ax = plt.subplots(2, 3)
                img_masked = sketch_gt_overlay(
                    image[0][grad_visualizer_modality].detach().numpy(),
                    label,
                    MODALITY[grad_visualizer_modality],
                )

                predicted_mask = sketch_gt_overlay(
                    image[0][grad_visualizer_modality].detach().numpy(),
                    predictions.detach().numpy(),
                    MODALITY[grad_visualizer_modality],
                )

                ax[0, 0].set_aspect(1)
                ax[0, 0].imshow(pos_grad[0], cmap='gray')
                ax[0, 0].axis("off")
                ax[0, 0].set_title("FLAIR")

                ax[0, 1].set_aspect(1)
                ax[0, 1].imshow(pos_grad[1], cmap='gray')
                ax[0, 1].axis("off")
                ax[0, 1].set_title("T1")

                ax[1, 0].set_aspect(1)
                ax[1, 0].imshow(pos_grad[2], cmap='gray')
                ax[1, 0].axis("off")
                ax[1, 0].set_title("T1-CE")

                ax[1, 1].set_aspect(1)
                ax[1, 1].imshow(pos_grad[3], cmap='gray')
                ax[1, 1].axis("off")
                ax[1, 1].set_title("T2")

                ax[0, 2].set_aspect(1)
                ax[0, 2].imshow(img_masked)
                ax[0, 2].axis("off")
                ax[0, 2].set_title(
                    "Image (" + MODALITY[grad_visualizer_modality] + ") + Mask"
                )
                black_patch = mpatches.Patch(color="black", label="BG")
                red_patch = mpatches.Patch(color="red", label="NET")
                green_patch = mpatches.Patch(color="green", label="ED")
                blue_patch = mpatches.Patch(color="blue", label="ET")
                ax[0, 2].legend(
                    handles=[black_patch, red_patch, green_patch, blue_patch],
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                )

                ax[1, 2].set_aspect(1)
                ax[1, 2].imshow(predicted_mask)
                ax[1, 2].axis("off")
                ax[1, 2].set_title("Predicted mask")

                plt.suptitle("Positive Gradients: "
                    + FLAG_TO_STUDY[flag_to_study]
                    + ", Class: "
                    + TARGET_CLASS_TO_STUDY[target_class_to_study], x=0.5, y=0.95)
                plt.tight_layout()
                plt.savefig(
                    image_output_path
                    + data["id"][0]
                    + "_positive_grad_flag_"
                    + FLAG_TO_STUDY[flag_to_study]
                    + "_class_"
                    + TARGET_CLASS_TO_STUDY[target_class_to_study]
                    + "_count_"
                    + str(count)
                    + ".pdf",
                    dpi=300,
                )
                if show:
                    plt.show()
                plt.close()


                fig, ax = plt.subplots(2, 3)
                img_masked = sketch_gt_overlay(
                    image[0][grad_visualizer_modality].detach().numpy(),
                    label,
                    MODALITY[grad_visualizer_modality],
                )

                predicted_mask = sketch_gt_overlay(
                    image[0][grad_visualizer_modality].detach().numpy(),
                    predictions.detach().numpy(),
                    MODALITY[grad_visualizer_modality],
                )

                ax[0, 0].set_aspect(1)
                ax[0, 0].imshow(neg_grad[0], cmap='gray')
                ax[0, 0].axis("off")
                ax[0, 0].set_title("FLAIR")

                ax[0, 1].set_aspect(1)
                ax[0, 1].imshow(neg_grad[1], cmap='gray')
                ax[0, 1].axis("off")
                ax[0, 1].set_title("T1")

                ax[1, 0].set_aspect(1)
                ax[1, 0].imshow(neg_grad[2], cmap='gray')
                ax[1, 0].axis("off")
                ax[1, 0].set_title("T1-CE")

                ax[1, 1].set_aspect(1)
                ax[1, 1].imshow(neg_grad[3], cmap='gray')
                ax[1, 1].axis("off")
                ax[1, 1].set_title("T2")

                ax[0, 2].set_aspect(1)
                ax[0, 2].imshow(img_masked)
                ax[0, 2].axis("off")
                ax[0, 2].set_title(
                    "Image (" + MODALITY[grad_visualizer_modality] + ") + Mask"
                )
                black_patch = mpatches.Patch(color="black", label="BG")
                red_patch = mpatches.Patch(color="red", label="NET")
                green_patch = mpatches.Patch(color="green", label="ED")
                blue_patch = mpatches.Patch(color="blue", label="ET")
                ax[0, 2].legend(
                    handles=[black_patch, red_patch, green_patch, blue_patch],
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                )

                ax[1, 2].set_aspect(1)
                ax[1, 2].imshow(predicted_mask)
                ax[1, 2].axis("off")
                ax[1, 2].set_title("Predicted mask")

                plt.suptitle("Negative Gradients: "
                    + FLAG_TO_STUDY[flag_to_study]
                    + ", Class: "
                    + TARGET_CLASS_TO_STUDY[target_class_to_study], x=0.5, y=0.95)
                plt.tight_layout()
                plt.savefig(
                    image_output_path
                    + data["id"][0]
                    + "_negative_grad_flag_"
                    + FLAG_TO_STUDY[flag_to_study]
                    + "_class_"
                    + TARGET_CLASS_TO_STUDY[target_class_to_study]
                    + "_count_"
                    + str(count)
                    + ".pdf",
                    dpi=300,
                )
                if show:
                    plt.show()
                plt.close()