import h5py
import numpy as np
from matplotlib import pyplot as plt
from fastmri.data import transforms as T
import fastmri
import cv2
from skimage.metrics import structural_similarity
import os
import glob
import optparse
from pathlib import Path

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option('-i', help='Base Test Folder', dest='test_dir', action='store')
    parser.add_option('-o', help='Output Prediction Folder', dest='output_dir', action='store')
    (opts, args) = parser.parse_args()
    test_dir = opts.test_dir
    output_dir = opts.output_dir
    if not test_dir.endswith('/'):
        test_dir = test_dir + "/"
    if not output_dir.endswith('/'):
        output_dir = output_dir + "/"

    for test_filename in glob.glob(test_dir+"*.h5"):
        for output_filename in glob.glob(output_dir+"*.h5"):
            if os.path.basename(test_filename) == os.path.basename(output_filename):

                print("Comparing "+test_filename+" with "+output_filename)

                hdf_test = h5py.File(test_filename,'r')
                hdf_output = h5py.File(output_filename,'r')

                dset_validation = hdf_test["kspace"][()]
                dset_validation_esc = hdf_test["reconstruction_esc"][()]
                dset_output = hdf_output["reconstruction"]

                slice_count = dset_validation.shape[0]
                print("Slice count", slice_count)

                assert(dset_validation.shape[0] == dset_validation_esc.shape[0] == dset_output.shape[0])

                #Figure size is in inches, because why not. Divide by 96 to get ~pixels
                figure_height = (dset_validation.shape[1] / 96) * slice_count

                plot_index = 1
                plt.figure(figsize=(20, figure_height), dpi=240)
                for slice_index in range(0, slice_count):

                    val_slice_kspace = dset_validation[slice_index]
                    val_slice_kspace_tensor = T.to_tensor(val_slice_kspace)
                    val_slice_image = fastmri.ifft2c(val_slice_kspace_tensor)
                    val_slice_image_abs = fastmri.complex_abs(val_slice_image)

                    val_slice_esc_tensor = dset_validation_esc[slice_index]

                    output_slice_tensor = dset_output[slice_index]

                    plt.subplot(slice_count, 4, plot_index)
                    plt.title('validation k-space')
                    plt.imshow(val_slice_image_abs, cmap='gray')
                    plot_index += 1

                    plt.subplot(slice_count, 4, plot_index)
                    plt.title('reconstruction_esc')
                    plt.imshow(val_slice_esc_tensor, cmap='gray')
                    plot_index += 1

                    plt.subplot(slice_count, 4, plot_index)
                    plt.title('Target Reconstruction')
                    plt.imshow(output_slice_tensor, cmap='gray')
                    plot_index += 1

                    score, diff = structural_similarity(val_slice_esc_tensor, output_slice_tensor, full=True)
                    print("Image similarity", score)
                    diff = (diff * 255).astype("uint8")

                    """
                    Adapted from: https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
                    """
                    threshold = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    mask = np.zeros(val_slice_esc_tensor.shape, dtype='uint8')
                    filled_after = output_slice_tensor.copy()

                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > 1:
                            x,y,w,h = cv2.boundingRect(c)
                            cv2.rectangle(val_slice_esc_tensor, (x, y), (x + w, y + h), (36,255,12), 2)
                            cv2.rectangle(output_slice_tensor, (x, y), (x + w, y + h), (36,255,12), 2)
                            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
                            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

                    plt.subplot(slice_count, 4, plot_index)
                    plt.title('Reconstruction Difference')
                    plt.imshow(diff, cmap='gray')
                    plot_index += 1
                    
                out_filename_only = Path(output_filename).stem
                plt.savefig('Output_Comparison_ComplexSpace_'+out_filename_only+'.png')