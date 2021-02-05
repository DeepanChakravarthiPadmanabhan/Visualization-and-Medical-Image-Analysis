import typing
import numpy as np
import gin
import torch
import torchvision
import cv2


def get_transform(transform_type: str, params: typing.Dict):
    transform_class = {
        "Resize": Resize,
        "RandomOrientation": RandomOrientation,
        "RandomCrop": RandomCrop,
        "NormalizeImage": NormalizeImage,
    }[transform_type]
    return transform_class(**params)


@gin.configurable
def configure_transforms(config: typing.Dict) -> torchvision.transforms:
    transforms_list = []
    for i in range(len(config)):
        transforms_list.append(get_transform(**config[str(i)]))
    transforms_list.append(ToTensor())
    transform = torchvision.transforms.Compose(transforms_list)
    return transform


class Resize(object):
    """Resize the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = tuple(output_size)

    def resize(self, image: np.ndarray) -> np.ndarray:
        new_h, new_w = self.output_size
        new_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return new_image

    def __call__(self, sample: typing.Dict) -> typing.Dict:

        sample["image"] = self.resize(sample["image"])

        if "label" in sample.keys():
            mask = sample["label"]
            new_mask = self.resize(mask)
            sample["label"] = new_mask

        return sample


class NormalizeImage(object):
    def __init__(self, mean: float, stddev: float):
        assert isinstance(mean, float)
        assert isinstance(stddev, float)
        self.mean = mean
        self.stddev = stddev

    def normalize(self, image: np.ndarray) -> np.ndarray:
        # normalize data
        for i in range(image.shape[2]):
            if image[:, :, i].max() > 0:
                image[:, :, i] = image[:, :, i] / image[:, :, i].max()
        return image

    def __call__(self, sample: typing.Dict) -> typing.Dict:

        if isinstance(sample["image"], list):
            normalized_images = []
            for i in sample["image"]:
                normalized_images.append(self.normalize(i))
            sample["image"] = normalized_images
        else:
            sample["image"] = self.normalize(sample["image"])
        return sample


class ToTensor(object):
    def convert_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Channels/ Modalities to the first dimension
        return image

    def __call__(self, sample: typing.Dict) -> typing.Dict:

        if isinstance(sample["image"], list):
            tensor_images = []
            for i in sample["image"]:
                tensor_images.append(self.convert_to_tensor(i))
            sample["image"] = tensor_images
        else:
            sample["image"] = self.convert_to_tensor(sample["image"])

        if "label" in sample.keys():
            if isinstance(sample["label"], list):
                label = []
                for i in sample["label"]:
                    i[np.where(i == 4)] = 3
                    i = torch.from_numpy(i).float()
                    label.append(i)
                sample["label"] = label
            else:
                # TODO: Check whether this label renaming is necessary
                sample["label"][np.where(sample["label"] == 4)] = 3
                label = torch.from_numpy(sample["label"]).float()
                one_hot_tensor = torch.nn.functional.one_hot(
                    label.long(), num_classes=4
                )
                sample["label"] = one_hot_tensor.permute(2, 0, 1).to(
                    dtype=torch.float32
                )

        return sample


class RandomOrientation(object):
    pass


class RandomCrop(object):
    pass
