import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
# List of 5 NIfTI file paths
nii_paths = [
    "Brats18_2013_2_1_flair.nii",
    "Brats18_2013_2_1_t1.nii",
    "Brats18_2013_2_1_t1ce.nii",
    "Brats18_2013_2_1_t2.nii",
    "Brats18_2013_2_1_seg.nii"
]

# Load all volumes
volumes = []
for path in nii_paths:
    data = nib.load(path).get_fdata()
    
    # Normalize each volume using cv2 to range [0, 255]
    norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    norm_uint8 = np.uint8(norm)
    
    volumes.append(norm_uint8)
# Get common depth range (minimum slices across volumes)
min_depth = min([v.shape[2] for v in volumes])

print(f"Volumes loaded. Common axial depth: 0 to {min_depth - 1}")

# gt_mask = (volumes[4] != 0).astype(np.uint8)
# values, counts = np.unique(gt_mask, return_counts=True)

# for v, c in zip(values, counts):
#     print(f"Giá trị {v}: {c} lần")
while True:
    try:
        user_input = input(f"\nEnter slice number (0–{min_depth - 1}, or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

        slice_idx = int(user_input)
        if 0 <= slice_idx < min_depth:
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            for i in range(5):
                axs[i].imshow(volumes[i][:, :, slice_idx], cmap="gray")
                axs[i].set_title(f"Image {i+1} - Slice {slice_idx}")
                axs[i].axis('off')
            plt.tight_layout()
            plt.show()
            pass
        else:
            print("⚠️ Slice index out of range.")
        
    except ValueError:
        print("⚠️ Please enter a valid number or 'q' to quit.")