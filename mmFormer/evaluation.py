import argparse
import os
import logging
import torch
import numpy as np
from predict import test_softmax, AverageMeter
from mmformer import Model
from data.datasets_nii import Brats_loadall_test_nii
from utils.parser import setup
from utils.lr_scheduler import MultiEpochsDataLoader

def evaluate_model():
    parser = argparse.ArgumentParser()
    # Required arguments from training
    parser.add_argument('--datapath', default=None, type=str)
    parser.add_argument('--dataname', default='BRATS2018', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    # Additional required arguments for setup
    parser.add_argument('--savepath', default='./evaluation_results', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--resume', default=None, type=str)
    args = parser.parse_args()

    # Create evaluation results directory
    os.makedirs(args.savepath, exist_ok=True)
    
    setup(args, 'evaluation')
    
    # Set up model
    num_cls = 4 if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018'] else 5
    model = Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    
    # Load model weights
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f'Loaded model from epoch: {checkpoint["epoch"]}')
    
    # Setup test data
    test_transforms = 'Compose([NumpyType((np.float32, np.int64, np.float32)),])'
    test_file = 'test3.txt' if args.dataname == 'BRATS2018' else 'test.txt'
    test_set = Brats_loadall_test_nii(transforms=test_transforms, 
                                     root=args.datapath,
                                     num_cls=4,
                                     test_file=test_file)
    
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Define masks
    masks = [[False, False, False, True], [False, True, False, False], 
             [False, False, True, False], [True, False, False, False]]
    mask_name = ['t2', 't1c', 't1', 'flair']
    
    # Evaluate
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info(f'Testing {mask_name[i]}')
            dice_score = test_softmax(
                test_loader,
                model,
                dataname=args.dataname,
                feature_mask=mask
            )
            test_score.update(dice_score)
        logging.info(f'Average scores: {test_score.avg}')

if __name__ == '__main__':
    evaluate_model()