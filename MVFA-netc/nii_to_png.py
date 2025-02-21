import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Ensure the figure's padding is set correctly
plt.rcParams['savefig.pad_inches'] = 0

# Paths
input_dir = r'C:\Users\user\mvfa\MVFA-AD\brats goat\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth'  # Base directory containing the NIfTI files
output_dir = r'C:\Users\user\mvfa\MVFA-AD\brats goat\PNGData'  # Directory to save PNG files

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Process all NIfTI files in the input directory and maintain folder structure
for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            try:
                # Construct the file path for the input NIfTI file
                nii_file_path = os.path.join(root, filename)
                
                # Load the NIfTI file
                img = nib.load(nii_file_path)
                
                # Get the data array from the NIfTI file
                data = img.get_fdata()
                print(f"Processing {nii_file_path} with shape {data.shape}")

                # Check if the third dimension is larger than 0
                if data.shape[2] == 0:
                    print(f"Skipping {nii_file_path} due to empty third dimension")
                    continue
                
                # Select the middle slice along the third dimension
                middle_slice = data[:, :, data.shape[2] // 2].T
                
                # Plot the slice
                plt.imshow(middle_slice, cmap='gray')
                plt.axis('off')
                
                # Create the corresponding output directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Save the figure as a PNG file
                output_filename = os.path.join(output_subdir, f'{os.path.splitext(filename)[0]}.png')
                plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                
                # Clear the current figure to avoid memory issues
                plt.clf()
                
                print(f'Successfully processed and saved {output_filename}')
                
            except Exception as e:
                # Print the error message for debugging
                print(f'Failed to process file {nii_file_path}: {e}')
                continue
