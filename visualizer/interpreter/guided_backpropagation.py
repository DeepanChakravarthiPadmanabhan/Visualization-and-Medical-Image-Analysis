import os
import gin
import logging
import torch
from torch.nn import ReLU
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from visualizer.utils.constants import MODALITY, FLAG_TO_STUDY, TARGET_CLASS_TO_STUDY
from visualizer.utils.post_processing import sketch_gt_overlay, post_process_gradient, get_positive_negative_saliency
from copy import deepcopy

LOGGER = logging.getLogger(__name__)


class GuidedBackprop():

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
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                name_of_layers.append(module)
        return name_of_layers

    def hook_layers(self):

        def gradient_hook_function(module, grad_in, grad_out):
            # Grad-in is the gradient with respect to the image pixels in the shape (1, 4, h, w)
            # Grad-out is the gradient with respect to the 1st conv layer
            self.gradients = grad_in[0]

        def logit_hook_function(module, grad_in, grad_out):
            self.logit_output = grad_out

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
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
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

    def generate_gradients(self, image, target_class=1, flag_to_study=4):
        model_output = self.model(image)
        # Zero gradients
        self.model.zero_grad()
        # Formulate the target for backpropagation
        gradient_to_propagate = torch.zeros(self.logit_output.shape)

        if flag_to_study == 1:
            # Research question 1
            LOGGER.info('Performing visualization study: %s', flag_to_study)
            logit_for_targetclass = torch.zeros(self.logit_output.shape)
            logit_for_targetclass[0, target_class, ::] = self.logit_output[0, target_class, ::]
            gradient_to_propagate = logit_for_targetclass

        elif flag_to_study == 2:
            # Research question 2
            LOGGER.info('Performing visualization study: %s', flag_to_study)
            predictions = (model_output > 0.5).float()
            logit_for_classified_targetclass = torch.zeros(self.logit_output.shape)
            logit_for_classified_targetclass[0, target_class, ::] = predictions[0, target_class, ::]
            gradient_to_propagate = logit_for_classified_targetclass

        elif flag_to_study == 3:
            # Research question 3
            LOGGER.info('Performing visualization study: %s', flag_to_study)
            pass

        else:
            raise NotImplementedError('Only 3 visualization options available')

        if gradient_to_propagate.max() == torch.tensor(0):
            gradient_to_propagate += 1e-10
        # Propagate the logit information to the input
        model_output.backward(gradient=gradient_to_propagate)
        # Convert PyTorch variable to numpy array
        # [0] to get rid of the first channel (1, 4, 224, 224)
        gradients_as_arr = self.gradients.data.numpy()[0]

        return gradients_as_arr


@gin.configurable
def guided_backpropagate(model_path,
                         dataloader,
                         net,
                         report_output_path,
                         image_output_path,
                         visualize,
                         target_class_to_study = 1,
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

    if visualize and (
        not os.path.exists(os.path.join(image_output_path))
    ):
        os.makedirs(os.path.join(image_output_path))
        LOGGER.info(
            "Saving images in the directory: %s",
            os.path.join(image_output_path),
        )

    test_loader = dataloader[2]
    device = torch.device("cpu")

    state_test = torch.load(model_path, map_location=device)
    net.load_state_dict(state_test)
    GBP = GuidedBackprop(net)

    for data in test_loader:
        count = 0
        for image, label in zip(data["image"][idx], data["label"][idx]):
            count += 1
            LOGGER.info("Predicting on patient id: %s", data["id"][0])

            image.requires_grad_(True)
            image = image.unsqueeze(0)
            grad = GBP.generate_gradients(image, target_class=target_class_to_study, flag_to_study=flag_to_study)
            grad_for_visualization = grad[grad_visualizer_modality]
            processed_grad = post_process_gradient(grad_for_visualization)

            if visualize:

                fig, (ax1, ax2) = plt.subplots(1, 2)

                img_masked = sketch_gt_overlay(image[0][grad_visualizer_modality].detach().numpy(), label, MODALITY[grad_visualizer_modality])
                ax1.set_aspect(1)
                ax1.imshow(img_masked)
                ax1.axis('off')
                ax1.set_title('Image ('+ MODALITY[grad_visualizer_modality] + ') + Mask')
                black_patch = mpatches.Patch(color='black', label='BG')
                red_patch = mpatches.Patch(color='red', label='NET')
                green_patch = mpatches.Patch(color='green', label='ED')
                blue_patch = mpatches.Patch(color='blue', label='ET')
                ax1.legend(handles=[black_patch, red_patch, green_patch, blue_patch], bbox_to_anchor=(1.05, 1), loc='upper left',
                           borderaxespad=0.)

                ax2.set_aspect(1)
                ax2.imshow(processed_grad)
                ax2.axis('off')
                ax2.set_title('Gradient')

                plt.suptitle('Gradients: ' + FLAG_TO_STUDY[flag_to_study] + ', Class: ' + TARGET_CLASS_TO_STUDY[target_class_to_study], x=0.5, y=0.84)
                plt.tight_layout()
                plt.savefig(
                    image_output_path
                    + data["id"][0] + '_mask_grad_flag_' + FLAG_TO_STUDY[flag_to_study] + '_class_' + TARGET_CLASS_TO_STUDY[target_class_to_study] + '_count_'+ str(count)
                    + ".pdf", dpi=300
                )
                if show:
                    plt.show()
                plt.close(fig)

                sketch_gt_overlay(image[0][grad_visualizer_modality].detach().numpy(),
                                  label,
                                  image_output_path + data["id"][0] + '_input_mask_' + str(count) + '.pdf',
                                  MODALITY[grad_visualizer_modality],
                                  True,
                                  show)

                if visualize_saliency:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                    pos_saliency, neg_saliency = get_positive_negative_saliency(grad_for_visualization)

                    ax1.set_aspect(1)
                    ax1.imshow(processed_grad)
                    ax1.axis('off')
                    ax1.set_title('Gradients - out/inp')

                    ax2.set_aspect(1)
                    ax2.imshow(pos_saliency) # Positive probability impact
                    ax2.axis('off')
                    ax2.set_title('Positive')

                    ax3.set_aspect(1)
                    ax3.imshow(neg_saliency) # Negative probability impact
                    ax3.axis('off')
                    ax3.set_title('Negative')

                    plt.suptitle('Positive and negative saliency maps', x=0.5, y=0.8)
                    plt.tight_layout()
                    plt.savefig(
                        image_output_path
                        + data["id"][0] + '_all_grad_' + str(count)
                        + ".pdf", dpi=300
                    )
                    if show:
                        plt.show()
                    plt.close()

