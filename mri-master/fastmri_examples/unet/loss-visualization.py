import h5py
import numpy as np
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T
from fastmri.evaluate import ssim, mse, nmse
import torch

original_data_dir = "../../../data"
model_data_dir_prefix = "unet/unet_demo_"
slice_num = 22

class ModelInfo:
    def __init__(self, label, dir):
        self.label = label
        self.dir = dir

def main():
    fig = plt.figure()
    # file_name = '1000001'
    # file_name = '1000005'
    # file_name = '1000070'
    # file_name = '1000583'
    # file_name = '1000695'
    file_name = '1000769'
    a = plot_ground_truth(file_name, 231)
    b = plot_reconstruction(model_info_dict['l1'], file_name, 232)
    c = plot_reconstruction(model_info_dict['l2'], file_name, 233)
    d = plot_reconstruction(model_info_dict['ssim'], file_name, 234)
    e = plot_reconstruction(model_info_dict['ms_ssim'], file_name, 235)
    f = plot_reconstruction(model_info_dict['l1_ssim_alpha_0.1'], file_name, 236)
    plt.savefig(f'diagrams/{file_name}-basic.png', bbox_inches='tight')
    plt.show()

def plot_ground_truth(file_name, position=111):
    hf = h5py.File(f'{original_data_dir}/singlecoil_test/file{file_name}.h5')
    plot_h5_file(hf, position, 'Ground Truth')
    plt.axis('off')
    return hf['reconstruction_esc']

def plot_reconstruction(model_info: ModelInfo, file_name, position=111):
    hf = h5py.File(f'{model_data_dir_prefix}{model_info.dir}/reconstructions/file{file_name}.h5')
    reconstruction = hf['reconstruction']
    plot(reconstruction[slice_num], position, cmap='gray', label=model_info.label)
    # plot_h5_file(hf, position, model_info.label)
    plt.axis('off')
    return reconstruction

def plot_h5_file(hf, position=111, label=''):
    # volume_kspace = hf['kspace'][()]
    # slice_kspace = volume_kspace[20]
    #
    # slice_kspace2 = T.to_tensor(slice_kspace)  # Convert from numpy array to pytorch tensor
    # slice_image = fastmri.ifft2c(slice_kspace2)  # Apply Inverse Fourier Transform to get the complex image
    # slice_image_abs = fastmri.complex_abs(slice_image)  # Compute absolute value to get a real image
    # slice_image_abs = T.center_crop(slice_image_abs, (320, 320))
    # target, mean, std = T.normalize_instance(slice_image_abs, eps=1e-11)
    # target = target.clamp(-6, 6)
    #
    # plot(target, position, label=label, cmap='gray')
    target = hf['reconstruction_esc'][slice_num]
    plot(target, position, label=label, cmap='gray')

def plot(data, position=111, cmap=None, label=''):
    ax = plt.subplot(position)
    ax.set_title(label)
    plt.imshow(data, cmap)

model_info_dict = {
    "l1": ModelInfo("L1", "l1_loss"),
    "l2": ModelInfo("L2", "l2_loss"),
    "ssim": ModelInfo("SSIM", "ssim_loss"),
    "ms_ssim": ModelInfo("MS-SSIM", "ms_ssim_loss"),
    "l1_ssim_alpha_0.1": ModelInfo("L1 SSIM alpha 0.1", "l1_ssim_loss_alpha_point_1"),
    "l1_ssim_alpha_0.5": ModelInfo("L1 SSIM alpha 0.5", "l1_ssim_loss_alpha_point_5"),
    "l1_ssim_alpha_0.9": ModelInfo("L1 SSIM alpha 0.9", "l1_ssim_loss_alpha_point_9"),
    "l1_ms_ssim_alpha_0.1": ModelInfo("L1 MS-SSIM alpha 0.1", "l1_ms_ssim_loss_alpha_point_1"),
    "l1_ms_ssim_alpha_0.5": ModelInfo("L1 MS-SSIM alpha 0.5", "l1_ms_ssim_loss_alpha_point_5"),
    "l1_ms_ssim_alpha_0.9": ModelInfo("L1 MS-SSIM alpha 0.9", "l1_ms_ssim_loss_alpha_point_9"),
    "l2_ssim_alpha_0.1": ModelInfo("L2 SSIM alpha 0.1", "l2_ssim_loss_alpha_point_1"),
    "l2_ssim_alpha_0.5": ModelInfo("L2 SSIM alpha 0.5", "l2_ssim_loss_alpha_point_5"),
    "l2_ssim_alpha_0.9": ModelInfo("L2 SSIM alpha 0.9", "l2_ssim_loss_alpha_point_9"),
    "l2_ms_ssim_alpha_0.1": ModelInfo("L2 MS-SSIM alpha 0.1", "l2_ms_ssim_loss_alpha_point_1"),
    "l2_ms_ssim_alpha_0.5": ModelInfo("L2 MS-SSIM alpha 0.5", "l2_ms_ssim_loss_alpha_point_5"),
    "l2_ms_ssim_alpha_0.9": ModelInfo("L2 MS-SSIM alpha 0.9", "l2_ms_ssim_loss_alpha_point_9")
}

if __name__ == "__main__":
    main()
