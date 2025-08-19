import torch
import os
# Set CUDA memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch.optim as optim
import data as Data
import models as Model
import torch.nn as nn
import argparse
import logging
import core.logger as Logger
import numpy as np
from misc.metric_tools import ConfuseMatrixMeter
from models.loss import *
from collections import OrderedDict
import core.metrics as Metrics
from misc.torchutils import get_scheduler, save_network
import wandb
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from itertools import islice

# =============================
# Run-naming, seed, and results
# =============================
def set_all_seeds(seed):
    import random
    import contextlib
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Parse experiment-level arguments early
parser = argparse.ArgumentParser(description='Cascade CD Training (early args parsing)')
parser.add_argument('--model', type=str, default='mamba', help='Model name for run naming')
parser.add_argument('--dataset', type=str, default='second', help='Dataset name for run naming')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--tag', type=str, default='', help='Optional custom tag for run naming')
early_args, unknown = parser.parse_known_args()

# Compose run name
now = datetime.now().strftime('%Y%m%d-%H')
run_name = f"{early_args.model}_{early_args.dataset}_{now}_{early_args.seed}"
if early_args.tag:
    run_name += f"_{early_args.tag}"

# Results folder convention - will be set after loading config
RESULTS_ROOT = None
RESULTS_DIR = None
LOG_DIR = None
CHECKPOINT_DIR = None

# Set all random seeds
set_all_seeds(early_args.seed)

# Initial run info
print(f"[INFO] Run name: {run_name}")
print(f"[INFO] Seed: {early_args.seed}")

def normalize_change_target(seg1: torch.Tensor | None,
                            seg2: torch.Tensor | None,
                            change_gt: torch.Tensor | None) -> torch.Tensor:
    """Return binary change target of shape [B, H, W] (dtype long, {0,1}).

    Handles cases where:
    - change_gt is provided in various formats (NCHW with C=1, NHWC RGB, or [B,H,W] int/float)
    - or must be derived from seg1 and seg2 which may be RGB (NHWC), one-hot/logits (NCHW, C>1),
      or single-channel (NCHW, C=1 or [B,H,W]).
    """
    def _to_index_mask(x: torch.Tensor) -> torch.Tensor:
        # Convert arbitrary segmentation label tensor to [B,H,W] integer indices
        if x.dim() == 4:
            # NCHW
            if x.size(1) == 1:
                return x.squeeze(1).long()
            # NHWC (e.g., RGB mask)
            if x.size(-1) == 3 and x.shape[1] != 3:
                # Assume channels-last
                return x.any(dim=-1).long()  # fallback to binary presence per-pixel
            # Multi-channel: take argmax as class indices
            return torch.argmax(x, dim=1).long()
        elif x.dim() == 3:
            # Already [B,H,W]
            return x.long()
        else:
            raise ValueError(f"Unsupported seg shape: {tuple(x.shape)}")

    if change_gt is not None:
        c = change_gt
        # If NHWC RGB -> any over last channel
        if c.dim() == 4 and c.size(-1) == 3 and (c.shape[1] != 3):
            c = c.any(dim=-1).long()
        elif c.dim() == 4 and c.size(1) == 1:
            c = c.squeeze(1).long()
        elif c.dim() == 3:
            # [B,H,W] possibly float/binary
            c = (c > 0).long()
        else:
            # As a conservative fallback
            c = _to_index_mask(c)
            c = (c > 0).long()
        return c

    # Derive from seg1 and seg2
    if seg1 is None or seg2 is None:
        raise ValueError("seg1/seg2 required when change_gt is None")

    # Handle RGB NHWC
    if seg1.dim() == 4 and seg1.size(-1) == 3 and (seg1.shape[1] != 3) and \
       seg2.dim() == 4 and seg2.size(-1) == 3 and (seg2.shape[1] != 3):
        change = (seg1 != seg2).any(dim=-1).long()
        return change

    # Convert to class indices if needed
    s1 = _to_index_mask(seg1)
    s2 = _to_index_mask(seg2)
    change = (s1 != s2).long()
    return change

def create_color_mask(tensor, num_classes: int = 10):
    """Convert a 2-D label tensor/ndarray to an RGB image with a categorical colormap.

    This is used for logging multi-class segmentation masks to wandb so that they
    appear in color instead of a binary/grayscale mask.
    """
    import numpy as _np
    import matplotlib as _mpl

    # Convert to numpy array
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = _np.asarray(tensor)

    # Remove singleton dimensions if they exist (e.g. 1×H×W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = _np.squeeze(arr, axis=0)
    
    # Handle case where ground truth is already RGB (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Already an RGB image, return as uint8
        return arr.astype(_np.uint8)
    
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mask or 3-D RGB image, got shape {arr.shape}")

    h, w = arr.shape
    unique_vals = _np.unique(arr)
    
    # Fix matplotlib deprecation warning and ensure class 0 is visible
    cmap = _mpl.colormaps.get_cmap('tab10')
    if hasattr(cmap, 'resampled'):
        cmap = cmap.resampled(num_classes)
    rgb = _np.zeros((h, w, 3), dtype=_np.uint8)
    
    # Custom color mapping to ensure class 0 is visible (not black)
    colors = []
    for i in range(num_classes):
        color = _np.array(cmap(i)[:3]) * 255
        # If color is too dark (close to black), make it brighter
        if _np.sum(color) < 50:  # Very dark color
            color = _np.array([255, 0, 0])  # Make it red instead
        colors.append(color.astype(_np.uint8))
    
    # Apply color mapping
    for cls in range(num_classes):
        if cls in unique_vals:
            rgb[arr == cls] = colors[cls]
    
    return rgb

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing',)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('-log_eval', action='store_true')
    # Accept naming-related args so CLI doesn't error (used only for run naming)
    parser.add_argument('--model', type=str, default=early_args.model, help='Model name (for run naming only)')
    parser.add_argument('--dataset', type=str, default=early_args.dataset, help='Dataset name (for run naming only)')
    parser.add_argument('--tag', type=str, default=early_args.tag, help='Optional custom tag (for run naming only)')
    # AMP controls
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision (AMP)')
    parser.add_argument('--amp_dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16'], help='AMP dtype selection')
    # Accept seed here as well (even though seeding uses early_args)
    parser.add_argument('--seed', type=int, default=None, help='Optional; accepted for compatibility')
    # Limits for overfitting/quick runs
    parser.add_argument('--max_train_batches', type=int, default=0, help='Limit number of training batches per epoch (0 = no limit)')
    parser.add_argument('--max_val_batches', type=int, default=0, help='Limit number of validation batches per epoch (0 = no limit)')
    parser.add_argument('--max_test_batches', type=int, default=0, help='Limit number of test batches (0 = no limit)')
    # Threshold for converting probs to binary mask (class-1)
    parser.add_argument('--change_threshold', type=float, default=0.5, help='Probability threshold for change class (class-1) binarization')

    # Parse config
    args = parser.parse_args()
    opt = Logger.parse(args)

    # Convert to NoneDict, which returns None for missing key
    opt = Logger.dict_to_nonedict(opt)
    
    # Use our run_name instead of generating a new timestamp
    # This ensures consistency between our early-parsed args and config
    exp_folder = run_name
    
    # Set up paths based on config and run name
    for k in ['log', 'result', 'checkpoint']:
        if k in opt['path_cd'] and isinstance(opt['path_cd'][k], str):
            base_dir = opt['path_cd'][k]
            stamped = os.path.join(base_dir, exp_folder)
            opt['path_cd'][k] = stamped
            os.makedirs(stamped, exist_ok=True)
            
            # Store paths in our global variables for reference
            if k == 'result':
                globals()['RESULTS_DIR'] = stamped
            elif k == 'log':
                globals()['LOG_DIR'] = stamped
            elif k == 'checkpoint':
                globals()['CHECKPOINT_DIR'] = stamped

    # =============================
    # AMP setup (default bf16 on A100/Ampere if available)
    # =============================
    amp_enabled = torch.cuda.is_available() and (not getattr(args, 'no_amp', False))
    if getattr(args, 'amp_dtype', 'auto') == 'bf16':
        amp_dtype = torch.bfloat16
    elif getattr(args, 'amp_dtype', 'auto') == 'fp16':
        amp_dtype = torch.float16
    else:
        # auto: prefer bf16 on SM>=80 (A100/Ampere+), else fp16
        try:
            cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
            amp_dtype = torch.bfloat16 if cc and cc[0] >= 8 else torch.float16
        except Exception:
            amp_dtype = torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    
    # Keep the subfolder name for reference
    opt['path_cd']['exp_folder'] = exp_folder
    
    # Log final path info now that we have the config
    print(f"[INFO] Results will be saved to: {RESULTS_DIR}")

    # Logging setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Set up file logging now that we have the log directory
    log_file = os.path.join(LOG_DIR, 'run.log')
    
    Logger.setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train',
                        level=logging.INFO, screen=True)
    Logger.setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test',
                        level=logging.INFO)
    logger = logging.getLogger('base')
    
    # Log experiment configuration and seed information
    logger.info(f"Run name: {run_name}")
    logger.info(f"Seed: {early_args.seed}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
    logger.info(Logger.dict2str(opt))

    # Set device with comprehensive debugging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # GPU Debugging Information
    logger.info(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA Device Count: {torch.cuda.device_count()}')
        logger.info(f'Current CUDA Device: {torch.cuda.current_device()}')
        logger.info(f'CUDA Device Name: {torch.cuda.get_device_name()}')
        logger.info(f'CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
        logger.info(f'CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
    else:
        logger.warning('CUDA is not available! Training will run on CPU (very slow)')

    # Initialize wandb only on main process
    if opt.get('wandb') and opt['wandb'].get('project'):
        wandb.init(project=opt['wandb']['project'], config=opt)
        try:
            run_name = exp_folder
            if hasattr(wandb, 'run') and wandb.run is not None:
                wandb.run.name = run_name
        except Exception:
            pass
    else:
        wandb.init(mode="disabled")

    #dataset
    for phase, dataset_opt in opt['datasets'].items(): #train train{}
        #print(" phase is {}, dataopt is {}".format(phase, dataset_opt))
        if phase == 'train' and args.phase != 'test':
            print("Creat [train] change-detection dataloader")
            train_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Creat [val] change-detection dataloader")
            val_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)

        # elif phase == 'test' and args.phase == 'test':
        elif phase == 'test':
            print("Creat [test] change-detection dataloader")
            test_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)

    logger.info('Initial Dataset Finished')

    #Create cd model
    cd_model = Model.create_CD_model(opt)
    
    # Initialize model weights to prevent NaN loss - more conservative
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight, gain=0.1)  # Very small gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.001)  # Very small std
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    cd_model.apply(init_weights)
    cd_model.to(device)
    logger.info(f'CD Model moved to device: {device}')
    
    # Verify model is actually on GPU
    if torch.cuda.is_available():
        model_device = next(cd_model.parameters()).device
        logger.info(f'Model parameters are on device: {model_device}')
        if model_device.type != 'cuda':
            logger.error('WARNING: Model parameters are NOT on GPU!')
        else:
            logger.info('✓ Model successfully moved to GPU')

    # Enable gradient checkpointing if available to save memory
    if hasattr(cd_model, 'gradient_checkpointing_enable'):
        cd_model.gradient_checkpointing_enable()

    # Set up binary change detection loss only (model outputs 2-channel logits)
    num_classes_change = 2
    logger.info("Configuring binary change detection loss (2 classes)")
    if opt['model']['loss'] == 'ce_dice':
        loss_fun_change = CEDiceLoss(num_classes=num_classes_change)
    elif opt['model']['loss'] == 'ce':
        loss_fun_change = cross_entropy_loss_fn
    elif opt['model']['loss'] == 'dice':
        loss_fun_change = DiceOnlyLoss(num_classes=num_classes_change)
    elif opt['model']['loss'] == 'ce2_dice1':
        loss_fun_change = CE2Dice1Loss(num_classes=num_classes_change)
    elif opt['model']['loss'] == 'ce1_dice2':
        loss_fun_change = CE1Dice2Loss(num_classes=num_classes_change)
    else:
        # Default to CE for unknown options
        logger.warning(f"Unknown loss '{opt['model']['loss']}', defaulting to cross-entropy for change detection.")
        loss_fun_change = cross_entropy_loss_fn

    # If loss is nn.Module, move to device
    if isinstance(loss_fun_change, nn.Module):
        loss_fun_change.to(device)

    #Create optimizer
    if opt['train']["optimizer"]["type"] == 'adam':
        optimer = optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    metric = ConfuseMatrixMeter(n_class=2)  # For binary change detection (change/no-change)
    log_dict = OrderedDict()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # if you really want this
    # Remove any empty_cache()/synchronize() calls outside diagnostics

    #################
    # Training loop #
    #################
    if opt['phase'] == 'train':
        best_mF1 = 0.0
        epoch_losses = []
        for current_epoch in range(0, opt['train']['n_epoch']):
            print("......Begin Training......")
            metric.clear()
            cd_model.train()
            train_result_path = '{}/train/{}'.format(opt['path_cd']['result'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            #################
            #    Training   #
            #################
            message = 'lr: %0.7f\n \n' % optimer.param_groups[0]['lr']
            logger.info(message)

            epoch_loss = 0

            # Initial memory cleanup
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()

            # Reduce gradient accumulation for memory savings
            accumulation_steps = 2  # Effective batch size = 1 * 2 = 2
            
            # Set memory fraction to avoid fragmentation (more conservative)
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Apply optional cap on training batches
            _max_train = getattr(args, 'max_train_batches', 0) or 0
            _train_total = min(len(train_loader), _max_train) if _max_train > 0 else len(train_loader)
            _train_iter = islice(train_loader, _max_train) if _max_train > 0 else train_loader
            for current_step, train_data in enumerate(tqdm(_train_iter, total=_train_total, desc=f"Train {current_epoch}/{opt['train']['n_epoch']}")):
                # Aggressive memory cleanup at start of each step
                # torch.cuda.empty_cache()
                # torch.cuda.synchronize()
                
                # Move data to GPU manually
                train_im1 = train_data['A'].to(device)
                train_im2 = train_data['B'].to(device)
                # Robust label extraction and move to device
                # Segmentation labels are not used in this pipeline
                change = (train_data['change'] if 'change' in train_data else train_data['L']).to(device)

                # Forward pass (model should return binary change logits [B,2,H,W])
                if amp_enabled:
                    with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                        outputs = cd_model(train_im1, train_im2)
                else:
                    outputs = cd_model(train_im1, train_im2)
                # Some implementations may still return a tuple; extract the change logits
                if isinstance(outputs, tuple):
                    change_pred = None
                    for o in outputs:
                        if torch.is_tensor(o):
                            change_pred = o
                            break
                    if change_pred is None:
                        change_pred = outputs[0]
                else:
                    change_pred = outputs
                
                del train_im1, train_im2
                # torch.cuda.empty_cache()

                # Build binary change target directly from 'change' tensor
                change_bin = change
                if change_bin.dim() == 4 and change_bin.size(1) == 1:
                    change_bin = change_bin.squeeze(1)
                change_bin = change_bin.long().clamp(0, 1)
                if amp_enabled:
                    with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                        train_loss = loss_fun_change(change_pred, change_bin)
                        train_loss = train_loss / accumulation_steps
                else:
                    train_loss = loss_fun_change(change_pred, change_bin)
                    train_loss = train_loss / accumulation_steps
                loss_dict = {'change': train_loss.item()}
                
                # Convert logits to predicted masks for logging
                with torch.no_grad():
                    pred_change = torch.argmax(change_pred, dim=1)
                
                # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                if current_step == 0 and current_epoch % 1 == 0:
                    # Debug: Check prediction values
                    print(f"\n=== TRAINING PREDICTIONS DEBUG (Epoch {current_epoch}) ===")
                    print(f"pred_change shape: {pred_change.shape}, unique values: {torch.unique(pred_change[0])}")
                    print(f"change_pred shape: {change_pred.shape}, min: {change_pred.min():.4f}, max: {change_pred.max():.4f}")
                    
                    # Convert input images from normalized [-1, 1] to [0, 255] for visualization
                    img_t1 = Metrics.tensor2img(train_data['A'][0:1], out_type=np.uint8, min_max=(-1, 1))
                    img_t2 = Metrics.tensor2img(train_data['B'][0:1], out_type=np.uint8, min_max=(-1, 1))
                    
                    # No segmentation ground-truth visualizations; only change maps are logged
                    
                    # Prepare binary GT change as black/white image
                    train_gt_change_bw = ((change[0] > 0).float().detach().cpu().numpy() * 255).astype(np.uint8)
                    
                    # Prepare binary prediction change as black/white image
                    pred_change_binary = ((pred_change[0] > 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    
                    # Also log probability maps for debugging
                    change_probs = torch.softmax(change_pred[0], dim=0)
                    
                    # Create probability visualizations
                    change_prob = change_probs[1].detach().cpu().numpy()
                    
                    wandb.log({
                        # Input images
                        "train/input_t1": [wandb.Image(img_t1, caption="Input Image T1")],
                        "train/input_t2": [wandb.Image(img_t2, caption="Input Image T2")],
                        
                        "train/pred_change": [wandb.Image(pred_change_binary, caption="Pred Change (binary BW)")],
                        "train/pred_change_prob": [wandb.Image(change_prob, caption="Pred Change Class-1 Probability")],
                        
                        # Ground truths with consistent color mapping
                        "train/gt_change": [wandb.Image(train_gt_change_bw, caption="GT Change (binary BW)")],
                        
                        "global_step": current_epoch * len(train_loader) + current_step
                    })
                
                # Save change prediction for metrics before cleanup
                change_pred = change_pred.detach()  # [B, 2, H, W]
                change_gt = (change > 0).long().detach()  # Binary ground truth
                
                # Check for NaN loss before backward pass
                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    logger.warning(f"NaN/Inf loss detected at epoch {current_epoch}, step {current_step}. Skipping this batch.")
                    optimer.zero_grad()
                    continue
                
                # Backward with optional AMP scaling
                if use_scaler:
                    scaler.scale(train_loss).backward()
                else:
                    train_loss.backward()
                if current_step == 0 and current_epoch == 0:
                    torch.cuda.synchronize(); import time; t0=time.time()
                    # do backward() here
                    torch.cuda.synchronize(); print("backward time:", time.time()-t0, "s")

                
                if (current_step + 1) % accumulation_steps == 0 or (current_step + 1) == len(train_loader):
                    # Gradient clipping to prevent explosion
                    if use_scaler:
                        # Unscale before clipping
                        scaler.unscale_(optimer)
                        torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_norm=0.5)
                        scaler.step(optimer)
                        scaler.update()
                        optimer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_norm=0.5)
                        optimer.step()
                        optimer.zero_grad()
                    
                # Clean up memory after each batch (avoid double deletion)
                del change
                if 'pred_change' in locals():
                    del pred_change
                
                log_dict['loss'] = train_loss.item()
                log_dict['loss_change'] = loss_dict['change']
                epoch_loss += train_loss.item()

                # For metric, use argmax over 2-class change head
                # Probability-based binarization for better alignment with prob maps
                probs = torch.softmax(change_pred, dim=1)[:, 1, ...]  # class-1 probs
                thresh = getattr(args, 'change_threshold', 0.5)
                binary_pred = (probs >= thresh).int()
                
                # Ground truth already binary (saved above)
                gt_np = change_gt.cpu().numpy().astype(np.uint8)
                pred_np = binary_pred.cpu().numpy()



                current_score = metric.update_cm(pr=pred_np, gt=gt_np)
                log_dict['running_acc'] = current_score.item()
                wandb.log({'train_loss': train_loss.item(), 'train_running_acc': current_score.item()})

                # Logging with GPU monitoring
                if current_step % opt['train']['train_print_iter'] == 0:
                    gpu_memory_info = ""
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                        gpu_memory_info = f", GPU Memory: {gpu_memory_allocated:.2f}GB/{gpu_memory_cached:.2f}GB"
                    
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f%s\n' % (
                        current_epoch, opt['train']['n_epoch'], current_step, len(train_loader), train_loss.item(),
                        current_score.item(), gpu_memory_info)
                    logger.info(message)
                
                # Final cleanup of saved tensors
                del change_pred, change_gt, binary_pred

            ### Epoch Summary ###
            scores = metric.get_scores()
            epoch_acc = scores['mf1']
            # Compute average epoch loss
            avg_epoch_loss = (epoch_loss / len(train_loader)) if len(train_loader) > 0 else 0.0
            
            # Log training epoch summary
            wandb.log({
                'train/epoch_mF1': epoch_acc,
                'train/epoch_loss': avg_epoch_loss,
                'train/epoch_mIoU': scores.get('mIoU', 0),
                'train/epoch_OA': scores.get('OA', 0),
                # flat keys as requested
                'train_epoch_mf1': epoch_acc,
                'train_epoch_loss': avg_epoch_loss,
                'epoch': current_epoch
            })
            
            ### VALIDATION LOOP ###
            logger.info('Starting validation...')
            val_metric = ConfuseMatrixMeter(n_class=2)
            cd_model.eval()
            val_loss_total = 0.0
            val_steps = 0
            shape_mismatch_logged = False  # Flag to log shape mismatch only once per epoch
            
            with torch.no_grad():
                # Apply optional cap on validation batches
                _max_val = getattr(args, 'max_val_batches', 0) or 0
                _val_total = min(len(val_loader), _max_val) if _max_val > 0 else len(val_loader)
                _val_iter = islice(val_loader, _max_val) if _max_val > 0 else val_loader
                for val_step, val_data in enumerate(tqdm(_val_iter, total=_val_total, desc=f"Val {current_epoch}")):
                    val_img1 = val_data['A'].to(device)
                    val_img2 = val_data['B'].to(device)
                    
                    # Handle validation labels same as training
                    if 'L1' in val_data and 'L2' in val_data:
                        val_seg_t1 = val_data['L1'].to(device)
                        val_seg_t2 = val_data['L2'].to(device)
                    else:
                        val_seg_t1 = val_seg_t2 = val_data.get('L', torch.zeros_like(val_img1[:, :1])).to(device)
                    # Prefer provided change; otherwise derive from segs
                    val_change = val_data['change'].to(device) if 'change' in val_data else None
                    
                    # Forward pass
                    # Forward pass, model returns only change logits
                    outputs = cd_model(val_img1, val_img2)
                    if isinstance(outputs, tuple):
                        val_change_pred = None
                        for o in outputs:
                            if torch.is_tensor(o):
                                val_change_pred = o
                                break
                        if val_change_pred is None:
                            val_change_pred = outputs[0]
                    else:
                        val_change_pred = outputs
                    # Create binary ground truth for validation (robust [B,H,W])
                    val_change_bin = normalize_change_target(val_seg_t1, val_seg_t2, val_change)
                    # Ensure targets are long dtype with values {0,1}
                    val_change_bin = val_change_bin.long().clamp(0, 1)
                    # Use 2-class criterion
                    val_loss = loss_fun_change(val_change_pred, val_change_bin)
                    val_loss_total += val_loss.item()
                    val_steps += 1
                    # Predictions for metrics
                    val_probs = torch.softmax(val_change_pred.detach(), dim=1)[:, 1, ...]
                    thresh = getattr(args, 'change_threshold', 0.5)
                    val_binary_pred = (val_probs >= thresh).int()
                    # Ensure both arrays have the same shape for metric calculation
                    val_gt_np = val_change_bin.cpu().numpy().astype(np.uint8)
                    val_pred_np = val_binary_pred.cpu().numpy()
                    
                    # Handle potential shape mismatches
                    if val_gt_np.shape != val_pred_np.shape:
                        # If ground truth has extra dimensions, squeeze them
                        if val_gt_np.ndim > val_pred_np.ndim:
                            val_gt_np = val_gt_np.squeeze()
                        # If prediction has extra dimensions, squeeze them
                        elif val_pred_np.ndim > val_gt_np.ndim:
                            val_pred_np = val_pred_np.squeeze()
                        
                        # If still mismatched, resize to match using PyTorch interpolation
                        if val_gt_np.shape != val_pred_np.shape:
                            if not shape_mismatch_logged:
                                logger.info(f"Validation shape mismatch (expected): gt={val_gt_np.shape}, pred={val_pred_np.shape} - handling automatically")
                                shape_mismatch_logged = True
                            
                            # Handle different tensor formats
                            if val_gt_np.ndim == 4 and val_gt_np.shape[-1] == 3:  # NHWC format (channels last)
                                # Take first channel and remove channel dimension
                                val_gt_np = val_gt_np[..., 0]  # Shape: (N, H, W)
                            elif val_gt_np.ndim == 3 and val_pred_np.ndim == 3:
                                # Both are 3D, try to match shapes by interpolation
                                val_gt_tensor = torch.from_numpy(val_gt_np).float().unsqueeze(1)  # Add channel dim: (N, 1, H, W)
                                val_gt_resized = F.interpolate(val_gt_tensor, size=val_pred_np.shape[-2:], mode='nearest')
                                val_gt_np = val_gt_resized.squeeze(1).numpy().astype(np.uint8)  # Remove channel dim
                            
                            # Final shape check
                            if val_gt_np.shape != val_pred_np.shape:
                                logger.warning(f"Still mismatched after processing: gt={val_gt_np.shape}, pred={val_pred_np.shape}")
                                # As last resort, flatten both and take minimum length
                                min_size = min(val_gt_np.size, val_pred_np.size)
                                val_gt_np = val_gt_np.flatten()[:min_size].reshape(-1)
                                val_pred_np = val_pred_np.flatten()[:min_size].reshape(-1)
                    
                    # Update confusion matrix and get running mF1
                    val_running_acc = val_metric.update_cm(pr=val_pred_np, gt=val_gt_np)
                    
                    # Per-step validation logging
                    wandb.log({'val_loss': val_loss.item(), 'val_running_acc': val_running_acc.item()})
            
                    # Log validation visualizations for first batch of each epoch
                    if val_step == 0 and current_epoch % 1 == 0:
                        with torch.no_grad():
                            val_pred_change = torch.argmax(val_change_pred, dim=1)
                        
                        # Debug: Check validation prediction values
                        print(f"\n=== VALIDATION PREDICTIONS DEBUG (Epoch {current_epoch}) ===")
                        print(f"val_pred_change shape: {val_pred_change.shape}, unique values: {torch.unique(val_pred_change[0])}")
                        print(f"val_change_pred shape: {val_change_pred.shape}, min: {val_change_pred.min():.4f}, max: {val_change_pred.max():.4f}")
                        
                        # Convert input images from normalized [-1, 1] to [0, 255] for visualization
                        val_img_t1 = Metrics.tensor2img(val_data['A'][0:1], out_type=np.uint8, min_max=(-1, 1))
                        val_img_t2 = Metrics.tensor2img(val_data['B'][0:1], out_type=np.uint8, min_max=(-1, 1))
                        
                        # Create binary ground truth change visualization (black/white)
                        val_gt_change_bw = ((val_change_bin[0] > 0).float().detach().cpu().numpy() * 255).astype(np.uint8)
                        
                        # Create binary change prediction visualization (black/white)
                        val_pred_change_binary = ((val_pred_change[0] > 0).detach().cpu().numpy() * 255).astype(np.uint8)
                        
                        # Probability map for change class-1
                        val_change_probs = torch.softmax(val_change_pred[0], dim=0)
                        val_change_prob = val_change_probs[1].detach().cpu().numpy()
                        
                        wandb.log({
                            "val/input_t1": [wandb.Image(val_img_t1, caption="Val Input Image T1")],
                            "val/input_t2": [wandb.Image(val_img_t2, caption="Val Input Image T2")],
                            "val/pred_change": [wandb.Image(val_pred_change_binary, caption="Val Pred Change (binary BW)")],
                            "val/pred_change_prob": [wandb.Image(val_change_prob, caption="Val Pred Change Class-1 Probability")],
                            "val/gt_change": [wandb.Image(val_gt_change_bw, caption="Val GT Change (binary BW)")],
                            "global_step": current_epoch * len(train_loader) + len(train_loader)
                        })
                    
                    # Clean up validation tensors
                    del val_change_pred, val_binary_pred
            
            # Log validation epoch summary
            val_scores = val_metric.get_scores()
            val_epoch_acc = val_scores['mf1']
            avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
            
            wandb.log({
                'val/epoch_loss': avg_val_loss,
                'val/epoch_mF1': val_epoch_acc,
                'val/epoch_mIoU': val_scores.get('mIoU', 0),
                'val/epoch_OA': val_scores.get('OA', 0),
                'epoch': current_epoch
            })
            
            logger.info(f'Validation - Epoch: {current_epoch}, Loss: {avg_val_loss:.5f}, mF1: {val_epoch_acc:.5f}')
            
            # Reset model to training mode
            cd_model.train()
            
            # Load the best model for testing
            gen_path = os.path.join(opt['path_cd']['checkpoint'], 'best_net.pth')
            if os.path.exists(gen_path):
                cd_model.load_state_dict(torch.load(gen_path), strict=True)
                logger.info(f'Loaded best model from {gen_path}')
            else:
                logger.warning(f'Best model not found at {gen_path}, using current model')
            cd_model.to(device)
            metric.clear()
            cd_model.eval()
            
            # Create test result directory
            test_result_path = '{}/test'.format(opt['path_cd']['result'])
            os.makedirs(test_result_path, exist_ok=True)
            
            with torch.no_grad():
                # Apply optional cap on test batches
                _max_test = getattr(args, 'max_test_batches', 0) or 0
                _test_total = min(len(test_loader), _max_test) if _max_test > 0 else len(test_loader)
                _test_iter = islice(test_loader, _max_test) if _max_test > 0 else test_loader
                for current_step, test_data in enumerate(tqdm(_test_iter, total=_test_total, desc="Test")):
                    test_img1 = test_data['A'].to(device)
                    test_img2 = test_data['B'].to(device)
                    # Robust label extraction - data automatically on correct device
                    if 'L1' in test_data and 'L2' in test_data:
                        seg_t1 = test_data['L1']
                        seg_t2 = test_data['L2']
                    else:
                        # Fallback for older single-label format
                        seg_t1 = seg_t2 = test_data.get('L')

                    # Obtain change mask if provided; otherwise derive binary mask from seg labels
                    change = test_data['change'] if 'change' in test_data else None

                    outputs = cd_model(test_img1, test_img2)
                    # Extract change logits robustly
                    if isinstance(outputs, tuple):
                        change_pred = None
                        for o in outputs:
                            if torch.is_tensor(o):
                                change_pred = o
                                break
                        if change_pred is None:
                            change_pred = outputs[0]
                    else:
                        change_pred = outputs
                    # Only use change head for metric and visuals (2-class)
                    # Convert prediction to binary change mask directly
                    probs = torch.softmax(change_pred.detach(), dim=1)[:, 1, ...]
                    thresh = getattr(args, 'change_threshold', 0.5)
                    G_pred = (probs >= thresh).int()
                    # Normalize GT to binary [B,H,W]
                    test_change_bin = normalize_change_target(seg_t1, seg_t2, change)
                    # Ensure targets are long dtype with values {0,1}
                    test_change_bin = test_change_bin.long().clamp(0, 1)

                    # Prepare numpy arrays for metrics
                    pred_np = G_pred.int().cpu().numpy()
                    gt_np = test_change_bin.cpu().numpy().astype(np.uint8)

                    # Optional: log first batch of test predictions (change + prob)
                    if current_step == 0:
                        # Convert input images from normalized [-1, 1] to [0, 255] for visualization
                        test_img_t1 = Metrics.tensor2img(test_data['A'][0:1], out_type=np.uint8, min_max=(-1, 1))
                        test_img_t2 = Metrics.tensor2img(test_data['B'][0:1], out_type=np.uint8, min_max=(-1, 1))
                        
                        # Change probabilities (class-1 probability)
                        change_probs = torch.softmax(change_pred[0], dim=0)
                        change_prob = change_probs[1].detach().cpu().numpy()
                        
                        # Create binary ground truth change (black=0, white=255)
                        test_gt_change_bw = ((test_change_bin[0] > 0).float().cpu().numpy() * 255).astype(np.uint8)
                        
                        # Create binary change prediction (black=0, white=255)
                        test_pred_change_binary = ((G_pred[0] > 0).detach().cpu().numpy() * 255).astype(np.uint8)
                        
                        wandb.log({
                            # Input images
                            "test/input_t1": [wandb.Image(test_img_t1, caption="Test Input Image T1")],
                            "test/input_t2": [wandb.Image(test_img_t2, caption="Test Input Image T2")],
                            
                            # Predictions
                            "test/pred_change": [wandb.Image(test_pred_change_binary, caption="Test Pred Change (binary BW)")],
                            # Probability map for change
                            "test/pred_change_prob": [wandb.Image(change_prob, caption="Test Pred Change Class-1 Probability")],
                            # Ground truth change
                            "test/gt_change": [wandb.Image(test_gt_change_bw, caption="Test GT Change (binary BW)")]
                        })
                    binary_pred = G_pred.int()
                    
                    # Get ground truth
                    if 'change' in test_data:
                        gt = (test_data['change'] > 0).long().to(device)
                    elif change is not None:
                        gt = change.to(device)
                    else:
                        gt = (seg_t1 != seg_t2).long().to(device)
                    
                    # Create binary ground truth for visualization
                    gt_binary = (gt > 0).int()  # Convert to binary (0 or 1)

                    # Visuals
                    out_dict = OrderedDict()
                    out_dict['pred_cm'] = binary_pred  # Use binary prediction for visualization
                    out_dict['gt_cm'] = gt_binary  # Use binary ground truth for visualization
                    visuals = out_dict

                    img_mode = 'single'
                    if img_mode == 'single':
                        # Binary masks already in {0,1}; keep range [0,1] for correct visualization
                        img_A = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        # Handle tensor dimensions properly for visualization
                        gt_tensor = visuals['gt_cm']
                        pred_tensor = visuals['pred_cm']
                        
                        # Ensure tensors are in correct format (B, H, W) before adding channel dimension
                        if gt_tensor.dim() > 3:
                            gt_tensor = gt_tensor.squeeze()  # Remove extra dimensions
                        if pred_tensor.dim() > 3:
                            pred_tensor = pred_tensor.squeeze()  # Remove extra dimensions
                            
                        # Add channel dimension and repeat for RGB
                        if gt_tensor.dim() == 3:  # (B, H, W)
                            gt_tensor = gt_tensor.unsqueeze(1)  # (B, 1, H, W)
                        elif gt_tensor.dim() == 2:  # (H, W)
                            gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                            
                        if pred_tensor.dim() == 3:  # (B, H, W)
                            pred_tensor = pred_tensor.unsqueeze(1)  # (B, 1, H, W)
                        elif pred_tensor.dim() == 2:  # (H, W)
                            pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                        
                        # Ensure float type for scaling
                        gt_tensor = gt_tensor.float()
                        pred_tensor = pred_tensor.float()
                        gt_cm = Metrics.tensor2img(gt_tensor.repeat(1, 3, 1, 1), out_type=np.uint8,
                                                   min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(pred_tensor.repeat(1, 3, 1, 1),
                                                     out_type=np.uint8, min_max=(0, 1))  # uint8

                        # Save imgs
                        Metrics.save_img(
                            img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))
                    else:
                        # grid img (keep masks in [0,1])
                        grid_img = torch.cat((test_data['A'],
                                              test_data['B'],
                                              visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                              visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                             dim=0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_{}.png'.format(test_result_path, current_step))

                ### log epoch status ###
                scores = metric.get_scores()
                epoch_acc = scores['mf1']
                log_dict['epoch_acc'] = epoch_acc.item()
                for k, v in scores.items():
                    log_dict[k] = v
                logs = log_dict
                message = '[Test CD summary]: Test mF1=%.5f \n' % \
                          (logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    message += '\n'
                logger.info(message)
                logger.info('End of testing...')

