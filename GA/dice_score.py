import nibabel as nib
import numpy as np

def dice_score(pred, target, epsilon=1e-6):
    min_dice = 1
    max_dice = 0
    min_i = 0
    max_i = 0
    print(pred.shape)
    
    for i in range(154):

        pred_2d = pred[:,:,i].astype(bool)
        target_2d = target[:,:,i].astype(bool)
    
    # pred = pred.astype(bool)
    # target = target.astype(bool)
    # values, counts = np.unique(pred, return_counts=True)

    # for v, c in zip(values, counts):
    #     print(f"Giá trị {v}: {c} lần")

    # values, counts = np.unique(target, return_counts=True)

    # for v, c in zip(values, counts):
    #     print(f"Giá trị {v}: {c} lần")
    
        intersection = np.logical_and(pred_2d, target_2d).sum()
        total = pred_2d.sum() + target_2d.sum()
        dice_score = (2. * intersection + epsilon) / (total + epsilon)
        if dice_score < min_dice:
            min_dice = dice_score
            min_i = i
        if dice_score > max_dice and dice_score != 1:
            max_dice = dice_score
            max_i = i
    print(f"min_dice: {min_dice}")
    print(f"min_i: {min_i}")
    print(f"max_dice: {max_dice}")
    print(f"max_i: {max_i}")
    return max_dice

# Load BraTS ground truth segmentation
gt_path = "Brats18_2013_2_1_seg.nii"
gt_nifti = nib.load(gt_path)
gt_data = gt_nifti.get_fdata()

# Create binary mask for Whole Tumor (labels 1, 2, 4)
gt_mask = (gt_data != 0).astype(np.uint8)
# Load BraTS ground truth segmentation
pred_path = "Brats18_2013_2_1_out.nii"
pred_nifti = nib.load(pred_path)
pred_mask = pred_nifti.get_fdata()

# Calculate Dice score
dice = dice_score(pred_mask, gt_mask)
print(f"Dice Score (Whole Tumor): {dice:.4f}")
import numpy as np
import matplotlib.pyplot as plt

class MaskDiffViewer:
    def __init__(self, pred, gt):
        self.pred = pred.astype(bool)
        self.gt = gt.astype(bool)
        self.slice_index = pred.shape[2] // 2
        self.max_slice = pred.shape[2] - 1

        self.false_positive = np.logical_and(self.pred, ~self.gt)
        self.false_negative = np.logical_and(~self.pred, self.gt)
        self.correct = np.logical_and(self.pred, self.gt)

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_display()

        plt.title('← → để chuyển slice | Trắng: đúng, Đỏ: FN, Xanh: FP')
        plt.axis('off')
        plt.show()

    def update_display(self):
        fp = self.false_positive[:, :, self.slice_index]
        fn = self.false_negative[:, :, self.slice_index]
        correct = self.correct[:, :, self.slice_index]

        vis = np.zeros(fp.shape + (3,), dtype=np.uint8)
        vis[correct] = [255, 255, 255]
        vis[fp] = [0, 255, 0]
        vis[fn] = [255, 0, 0]

        self.ax.clear()
        self.ax.imshow(vis)
        self.ax.set_title(f'Slice {self.slice_index} | ← → để điều hướng')
        self.ax.axis('off')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'left':
            self.slice_index = max(0, self.slice_index - 1)
            self.update_display()
        elif event.key == 'right':
            self.slice_index = min(self.max_slice, self.slice_index + 1)
            self.update_display()

# 🧪 Gọi viewer với 2 mask nhị phân cùng shape
viewer = MaskDiffViewer(pred_mask, gt_mask)