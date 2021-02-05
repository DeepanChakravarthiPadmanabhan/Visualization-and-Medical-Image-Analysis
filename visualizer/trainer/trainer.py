import os
import timeit
import logging
import gin
import typing
from sys import maxsize
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from visualizer.loss_function.dice_loss import DiceLoss


LOGGER = logging.getLogger(__name__)


@gin.configurable
class Trainer:
    """
    Trainer class. This class is used to define all the training parameters and processes for training the network.
    """

    def __init__(
        self,
        net,
        segment_criterion,
        optimizer_class,
        train_loader,
        valid_loader,
        model_output_path,
        input_height,
        input_width,
        secondary_loss=None,
        lr_step_size=5,
        lr=1e-04,
        patience=5,
        weight_decay=0,
        resume=False,
        num_epochs=50,
        scheduler=None,
        summary_writer=SummaryWriter(),
        device="cpu",
    ):
        self.net = net
        self.criterion_segmentation = segment_criterion()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_output_path = model_output_path
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.resume = resume
        self.num_epochs = num_epochs
        if secondary_loss == None:
            self.secondary_loss = secondary_loss
        else:
            self.secondary_loss = secondary_loss()
        self.optimizer = optimizer_class(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=0.1)
        self.tensorboard_writer = summary_writer
        self.device = device
        self.loss_scale = 1.0
        self.input_height = input_height
        self.input_width = input_width

        if self.device.type == "cuda":
            self.criterion_segmentation.cuda()

        if self.device.type == "cuda" and self.secondary_loss:
            self.secondary_loss.cuda()

    def _sample_to_device(self, sample):
        device_sample = {}
        for key, value in sample.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    device_sample[key] = value.to(self.device)
                elif isinstance(value, list):
                    device_sample[key] = [
                        t.to(self.device) if isinstance(t, torch.Tensor) else t
                        for t in value
                    ]
                else:
                    device_sample[key] = value

        return device_sample

    def train(self):

        LOGGER.info("Train Module")
        if not os.path.exists(os.path.dirname(self.model_output_path)):
            LOGGER.info(
                "Output directory does not exist. Creating directory %s",
                os.path.dirname(self.model_output_path),
            )
            os.makedirs(os.path.dirname(self.model_output_path))
        model_path = os.path.join(self.model_output_path, "model.pth")
        best_model_path = model_path
        patience_count = self.patience
        train_len = len(self.train_loader.batch_sampler)

        LOGGER.info("Ready to start training")
        tic = timeit.default_timer()
        best_validation_loss = maxsize

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            sample_size = 10
            self.net.train(True)

            running_loss = 0.0
            av_loss = 0.0

            for batch, data in enumerate(self.train_loader):
                LOGGER.info("Training: batch %d of epoch %d", batch + 1, epoch + 1)
                data = self._sample_to_device(data)

                input_image = data["image"]
                target = data["label"]

                if not isinstance(self.criterion_segmentation, DiceLoss):
                    # One hot encoding for loss functions like CE and focal loss
                    target = torch.argmax(target, dim=1)

                if not isinstance(self.secondary_loss, DiceLoss):
                    # One hot encoding for loss functions like CE and focal loss
                    secondary_target = torch.argmax(target, dim=1)

                self.optimizer.zero_grad()

                outputs_segmentation = self.net(input_image)

                loss1 = self.criterion_segmentation(outputs_segmentation, target)

                if self.secondary_loss:
                    loss2 = self.secondary_loss(outputs_segmentation, secondary_target)
                else:
                    loss2 = torch.tensor(0)

                LOGGER.info("Dice Loss: %s, CE" " Loss: %s", loss1, loss2)
                loss = loss1 + loss2
                loss.backward()

                self.optimizer.step()

                av_loss += loss.item() / self.loss_scale
                running_loss += loss.item() / self.loss_scale

                if batch % sample_size == (sample_size - 1):
                    LOGGER.info(
                        "epoch: %d, step: %d, loss: %f",
                        epoch + 1,
                        batch + 1,
                        running_loss / sample_size,
                    )
                    running_loss = 0.0

            if self.scheduler:
                self.scheduler.step()

            # output training loss
            av_train_loss = av_loss / train_len

            LOGGER.info(
                "epoch: %d, average loss: %f",
                epoch + 1,
                av_train_loss,
            )

            av_valid_loss = self.validation()

            LOGGER.info(
                "epoch: %d, average validation loss: %f",
                epoch + 1,
                av_valid_loss,
            )
            self.tensorboard_writer.add_scalar(
                "training/average_train_loss", av_train_loss, epoch + 1
            )
            self.tensorboard_writer.add_scalar(
                "validation/average_valid_loss", av_valid_loss, epoch + 1
            )

            if av_valid_loss < best_validation_loss and model_path and best_model_path:
                best_validation_loss = av_valid_loss
                best_model_path = model_path
                LOGGER.info(
                    "Better model found. Saving. Epoch %d, Path %s",
                    epoch + 1,
                    best_model_path,
                )
                torch.save(self.net.state_dict(), best_model_path)
                patience_count = self.patience
            else:
                patience_count -= 1
                LOGGER.info(
                    "No better model found. Epoch %d, Patience left %d",
                    epoch + 1,
                    patience_count,
                )

            if patience_count == 0:
                LOGGER.info(
                    "Epoch %d Patience is 0. Early stopping triggered", epoch + 1
                )
                break

        toc = timeit.default_timer()
        LOGGER.info("Finished training in %f s", toc - tic)

        self.tensorboard_writer.close()

    def write_validation_images(
        self,
        outputs_segmentation: torch.Tensor,
        target: torch.Tensor,
    ) -> typing.List:
        """
        Method to generate a list of figure for visualizing in the tensorboard.

        Args:
            outputs_segmentation: torch.Tensor
                Output prediction tensors for a validation data batch
            target:
                Target tensors for a validation data batch
        Returns:
            list_of_figs: List
                List of matplotlib figs with target and model predictions alongside

        """

        list_of_figs = []
        predictions = torch.argmax(outputs_segmentation, dim=1)

        for i in range(outputs_segmentation.shape[0]):
            prediction_per_batch = predictions[i]
            target_per_batch = target[i]
            fig = plt.figure(figsize=(32, 32))
            plt.subplot(121)
            plt.imshow(torch.squeeze(prediction_per_batch).cpu().detach().numpy())
            plt.subplot(122)
            plt.imshow(torch.squeeze(target_per_batch).cpu().detach().numpy())
            list_of_figs.append(fig)

        return list_of_figs

    def validation(self):
        LOGGER.info("Validation Module")
        valid_len = len(self.valid_loader.batch_sampler)

        self.net.train(False)

        av_loss = 0.0

        for batch, data in enumerate(self.valid_loader):
            data = self._sample_to_device(data)

            input_image = data["image"]
            target = data["label"]
            target = data["label"]
            secondary_target = torch.argmax(target, dim=1)

            self.optimizer.zero_grad()

            outputs_segmentation = self.net(input_image)

            if not isinstance(self.criterion_segmentation, DiceLoss):
                # One hot encoding for loss functions like CE and focal loss
                target = torch.argmax(target, dim=1)

            if self.secondary_loss:
                loss2 = self.secondary_loss(outputs_segmentation, secondary_target)
            else:
                loss2 = torch.tensor(0)

            loss1 = self.criterion_segmentation(outputs_segmentation, target)

            if self.secondary_loss:
                loss2 = self.secondary_loss(outputs_segmentation, secondary_target)
            else:
                loss2 = torch.tensor(0)

            loss = loss1 + loss2

            av_loss += loss.item() / self.loss_scale

            if batch == 0:
                list_of_figs = self.write_validation_images(
                    outputs_segmentation, secondary_target
                )
                self.tensorboard_writer.add_figure(
                    "Predictions vs target",
                    list_of_figs,
                    self.current_epoch * (batch + 1),
                )

        av_valid_loss = av_loss / valid_len
        return av_valid_loss
