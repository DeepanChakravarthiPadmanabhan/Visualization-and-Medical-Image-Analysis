import numpy as np
import nibabel as nib
import os
import re
import typing
import shutil


def get_doc_dirs_list(
    root_dir: str,
    images_save_path: str,
) -> typing.List:
    """

    Args:
        root_dir: string
        Directory containing the nii.gz files

        images_save_path: string
        Path to save the images in the directory containing nii.gz files

    Returns:
        image_dirs: list
        List containing directories of all the 4 modality nii.gz files

    """
    if os.path.exists(images_save_path):
        shutil.rmtree(images_save_path)
    os.makedirs(images_save_path)

    image_dirs = []

    for subdir, directory, files in os.walk(root_dir):
        image_directory = re.search(directory_pattern, subdir)
        if image_directory:
            image_dirs.append(subdir)

    if len(image_dirs) == 0:
        print("Root directory does not contain images. Kindly check the directory.")

    return image_dirs


def save_image_label(
    image_dir_list: typing.List,
    images_save_path: str,
) -> None:
    """

    Args:
        image_dirs: list
        List containing directories of all the 4 modality nii.gz files

        images_save_path: string
        Path to save the images in the directory containing nii.gz files

    """

    for image_dir in image_dir_list:
        directory_name = re.search(directory_pattern, image_dir)
        print("Saving directory: ", directory_name.group(1))
        t1 = os.path.join(image_dir, directory_name.group(1) + "_t1.nii.gz")
        t1ce = os.path.join(image_dir, directory_name.group(1) + "_t1ce.nii.gz")
        t2 = os.path.join(image_dir, directory_name.group(1) + "_t2.nii.gz")
        flair = os.path.join(image_dir, directory_name.group(1) + "_flair.nii.gz")
        seg = os.path.join(image_dir, directory_name.group(1) + "_seg.nii.gz")

        if (
            os.path.exists(t1)
            and os.path.exists(t1ce)
            and os.path.exists(t2)
            and os.path.exists(flair)
            and os.path.exists(seg)
        ):
            t1_img = nib.load(t1)
            t1_data = t1_img.get_fdata()

            t1ce_img = nib.load(t1ce)
            t1ce_data = t1ce_img.get_fdata()

            t2_img = nib.load(t2)
            t2_data = t2_img.get_fdata()

            flair_img = nib.load(flair)
            flair_data = flair_img.get_fdata()

            seg_img = nib.load(seg)
            seg_data = seg_img.get_fdata()

            for n, i in enumerate(range(t1_data.shape[2])):

                images_save_path_inner = (
                    images_save_path
                    + directory_name.group(1)
                    + "_slice_"
                    + str(n)
                    + "/"
                )
                if os.path.exists(images_save_path_inner):
                    shutil.rmtree(images_save_path_inner)
                os.makedirs(images_save_path_inner)

                t1_save = t1_data[:, :, i]
                np.save(
                    images_save_path_inner + directory_name.group(1) + "_t1", t1_save
                )

                t1ce_save = t1ce_data[:, :, i]
                np.save(
                    images_save_path_inner + directory_name.group(1) + "_t1ce",
                    t1ce_save,
                )

                t2_save = t2_data[:, :, i]
                np.save(
                    images_save_path_inner + directory_name.group(1) + "_t2", t2_save
                )

                flair_save = flair_data[:, :, i]
                np.save(
                    images_save_path_inner + directory_name.group(1) + "_flair",
                    flair_save,
                )

                seg_save = seg_data[:, :, i]
                np.save(
                    images_save_path_inner + directory_name.group(1) + "_seg", seg_save
                )


if __name__ == "__main__":

    # Define the train and test folders containing nii.gz files
    root_dir_train = "/Users/deepan/Documents/neuroscience_lab/trials_final/train/"
    root_dir_validation = (
        "/Users/deepan/Documents/neuroscience_lab/trials_final/validation/"
    )
    root_dir_test = "/Users/deepan/Documents/neuroscience_lab/trials_final/test/"

    # Define the path to save the train and test npy files
    images_save_path_train = (
        "/Users/deepan/Documents/neuroscience_lab/trials_final_save/train/"
    )
    images_save_path_validation = (
        "/Users/deepan/Documents/neuroscience_lab/trials_final_save/validation/"
    )
    images_save_path_test = (
        "/Users/deepan/Documents/neuroscience_lab/trials_final_save/test/"
    )

    if not os.path.exists(images_save_path_train):
        os.makedirs(images_save_path_train)
    else:
        shutil.rmtree(images_save_path_train)
        os.makedirs(images_save_path_train)
    if not os.path.exists(images_save_path_validation):
        os.makedirs(images_save_path_validation)
    else:
        shutil.rmtree(images_save_path_validation)
        os.makedirs(images_save_path_validation)
    if not os.path.exists(images_save_path_test):
        os.makedirs(images_save_path_test)
    else:
        shutil.rmtree(images_save_path_test)
        os.makedirs(images_save_path_test)

    # Directory pattern for BraTS dataset
    directory_pattern = re.compile("/(BraTS19_[A-Z0-9_]+)")

    image_dir_list_train = get_doc_dirs_list(root_dir_train, images_save_path_train)
    image_dir_list_validation = get_doc_dirs_list(
        root_dir_validation, images_save_path_validation
    )
    image_dir_list_test = get_doc_dirs_list(root_dir_test, images_save_path_test)
    save_image_label(image_dir_list_train, images_save_path_train)
    save_image_label(image_dir_list_validation, images_save_path_validation)
    save_image_label(image_dir_list_test, images_save_path_test)

# TODO: Clean this script and add automatic splitting to train-validation-test patient folders in the beginning
# TODO: Look into the dataset declaration in the train_dataset, test_dataset and valid_dataset. Not a good coding practice
# TODO: Look into PyTorch shuffle in dataloader
# TODO: Split patient ids to train, validation and test as done in the trials_final folder and split the entire dataset in Cluster
# TODO: Train the model for this new split in Cluster
