import argparse
import datetime
import os
import traceback
import logging
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm
import signal

from val import val
from backbone import HybridNetsBackbone
from utils.utils import get_last_weights, init_weights, boolean_string, \
    save_checkpoint, DataLoaderX, Params
from hybridnets.dataset import BddDataset
from hybridnets.custom_dataset import CustomDataset
from hybridnets.autoanchor import run_anchor
from hybridnets.model import ModelWithLoss
from utils.constants import *
from collections import OrderedDict
from torchinfo import summary


AMOUNT_TO_RUN_TRAIN_ON = 10000
AMOUNT_TO_RUN_VAL_ON = 3300





num_error_signals = 0
# Define a signal handler function
def signal_handler(sig, frame):
    signal_name = signal.Signals(sig).name
    logging.error(f"Received signal: {signal_name} ({signal})")
    # Perform any required actions or cleanup here
    # ...
    torch.cuda.empty_cache()
    global num_error_signals
    num_error_signals += 1
    if num_error_signals >= 2:
        exit(1)
    logging.info(f"Continuing training... num_error_signals = {num_error_signals}")

# Register the signal handler for keyboard interruption (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Register the signal handler for program termination signals
signal.signal(signal.SIGTERM, signal_handler)



def get_date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))


def init_logging(logdir):
    r"""
    Create log directory for storing checkpoints and output images.

    Args:
        logdir (str): Log directory name
    """
    logdir = os.path.join(logdir, get_date_uid())
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    log_file = os.path.join(logdir, 'log.log')
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    return logdir


def train(opt):
    torch.backends.cudnn.benchmark = True
    logging.info("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    params = Params(f'projects/{opt.project}.yml')

    if opt.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.log_path = init_logging(opt.log_path)
    opt.saved_path = opt.log_path + f'/checkpoints'
    pred_output_dir = opt.log_path + f'/predictions/'

    opt.log_path = opt.log_path + f'/{opt.project}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    os.makedirs(pred_output_dir, exist_ok=True)

    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

    logging.info("Loading train dataset")
    train_dataset = BddDataset(
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug,
        munit_output_path=opt.munit_path,
        amount_to_run_on=AMOUNT_TO_RUN_TRAIN_ON
    )

    training_generator = DataLoaderX(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    logging.info("Loading validation dataset")
    valid_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug,
        amount_to_run_on=AMOUNT_TO_RUN_VAL_ON
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    if params.need_autoanchor:
        params.anchors_scales, params.anchors_ratios = run_anchor(None, train_dataset)

    model = HybridNetsBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=opt.backbone,
                               seg_mode=seg_mode)

    # load last weights
    ckpt = {}
    # last_step = None
    if opt.load_weights:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        # try:
        #     last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        # except:
        #     last_step = 0

        try:
            ckpt = torch.load(weights_path)
            # new_weight = OrderedDict((k[6:], v) for k, v in ckpt['model'].items())
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        except RuntimeError as e:
            logging.info(f'[Warning] Ignoring {e}')
            logging.info(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        logging.info('[Info] initializing weights...')
        init_weights(model)

    logging.info('[Info] Successfully!!!')

    if opt.freeze_backbone:
        model.encoder.requires_grad_(False)
        model.bifpn.requires_grad_(False)
        logging.info('[Info] freezed backbone')

    if opt.freeze_det:
        model.regressor.requires_grad_(False)
        model.classifier.requires_grad_(False)
        model.anchors.requires_grad_(False)
        logging.info('[Info] freezed detection head')

    if opt.freeze_seg:
        model.bifpndecoder.requires_grad_(False)
        model.segmentation_head.requires_grad_(False)
        logging.info('[Info] freezed segmentation head')
    #summary(model, (1, 3, 384, 640), device='cpu')

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    model = model.to(memory_format=torch.channels_last)

    if opt.num_gpus > 0:
        model = model.cuda()

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)
    # logging.info(ckpt)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    # if opt.load_weights is not None and ckpt.get('optimizer', None):
        # scaler.load_state_dict(ckpt['scaler'])
        # optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    try:
        for epoch in range(opt.num_epochs):
            # last_epoch = step // num_iter_per_epoch
            # if epoch < last_epoch:
            #     continue
            logging.info(f'Epoch {epoch+1}/{opt.num_epochs}')
            
            epoch_loss = []
            progress_bar = tqdm(training_generator, ascii=True)
            for iter, data in enumerate(progress_bar):
                # if iter < step - last_epoch * num_iter_per_epoch:
                #     progress_bar.update()
                #     continue
                try:
                    imgs = data['img']
                    annot = data['annot']
                    seg_annot = data['segmentation']

                    if opt.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        imgs = imgs.to(device="cuda", memory_format=torch.channels_last)
                        annot = annot.cuda()
                        seg_annot = seg_annot.cuda()

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=opt.amp):
                        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                                seg_annot,
                                                                                                                obj_list=params.obj_list)
                        cls_loss = cls_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")
                        reg_loss = reg_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")
                        seg_loss = seg_loss.mean() if not opt.freeze_seg else torch.tensor(0, device="cuda")

                        loss = cls_loss + reg_loss + seg_loss
                        
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    writer.add_scalars('GPU_max_memory', {'train': torch.cuda.max_memory_allocated() / 1024 ** 3}, step)
                    writer.add_scalars('GPU_memory', {'train': torch.cuda.memory_allocated() / 1024 ** 3}, step)

                    scaler.scale(loss).backward()

                    # Don't have to clip grad norm, since our gradients didn't explode anywhere in the training phases
                    # This worsens the metrics
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()

                    # torch.cuda.empty_cache()

                    epoch_loss.append(float(loss))

                    if step % 100 == 0 and step > 0:
                        logging.info(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.3f}. Reg loss: {:.3f}. Seg loss: {:.3f}. Total loss: {:.3f}'.format(
                                step, epoch+1, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.float().item(),
                                reg_loss.item(), seg_loss.item(), loss.item()))
                        # writer.add_scalars('GPU_max_memory', {'train': torch.cuda.max_memory_allocated() / 1024 ** 3}, step)
                        # writer.add_scalars('GPU_memory', {'train': torch.cuda.memory_allocated() / 1024 ** 3}, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        writer.add_scalars('Loss', {'train': loss}, step)
                        writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                        writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                        writer.add_scalars('Segmentation_loss', {'train': seg_loss}, step)
                        
                        # log learning_rate
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('learning_rate', current_lr, step)
                        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
                        logging.info('Saved checkpoint to: ' + f'{opt.saved_path}/hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')

                except Exception as e:
                    logging.error('[Error]', traceback.format_exc())
                    logging.error(e)
                    continue

            scheduler.step(np.mean(epoch_loss))


            opt.cal_map = True if (epoch % opt.calc_mAP_interval == 0 and opt.cal_map) or epoch == opt.num_epochs else False  
            # if opt.cal_map is False, then it wont calculate mAP. 
            # it will always calculate mAP at the last epoch
            if epoch % opt.val_interval == 0:
                logging.info('Validating...')
                best_fitness, best_loss, best_epoch = val(model, val_generator, params, opt, seg_mode, tb_writer=writer, pred_output_dir=pred_output_dir,
                                                          is_training=True, optimizer=optimizer, scaler=scaler, writer=writer, epoch=epoch, step=step, 
                                                          best_fitness=best_fitness, best_loss=best_loss, best_epoch=best_epoch)
    except KeyboardInterrupt:
        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
    finally:
        writer.close()


def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Number of images per batch among all devices')
    parser.add_argument('--freeze_backbone', type=boolean_string, default=False,
                        help='Freeze encoder and neck (effnet and bifpn)')
    parser.add_argument('--freeze_det', type=boolean_string, default=False,
                        help='Freeze detection head')
    parser.add_argument('--freeze_seg', type=boolean_string, default=False,
                        help='Freeze segmentation head')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('--data_path', type=str, default='datasets/', help='The root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    #parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='Whether visualize the predicted boxes of training, '
                             'the output images will be in test/, '
                             'and also only use first 500 images.')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('-v', '--verbose', type=boolean_string, default=True,
                        help='Whether to print results per class when valing')
    parser.add_argument('--plots', type=boolean_string, default=True,
                        help='Whether to plot confusion matrix when valing')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to be used (0 to use CPU)')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                        help='Confidence threshold in NMS')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                        help='IoU threshold in NMS')
    parser.add_argument('--amp', type=boolean_string, default=False,
                        help='Automatic Mixed Precision training')
    # munit output path
    parser.add_argument('--munit_path', type=str, required=False, default=None)
    parser.add_argument('--calc_mAP_interval', type=int, default=10, help='Number of epoches between calculating mAP')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """

    tried to increase conf_thres to 0.5, because of issue (too high RAM memory, not related to GPU): https://github.com/datvuthanh/HybridNets/issues/44, 
        Still got killed at epoch 15.
    Tried --cal_map False, result: stopped at epoch 37
    Tried no validating, result: stopped at epoch 37
    Tried adding signal with --cal_map False, and trained only once at a time, result: WORKED!
    Tried adding signal with --cal_map False, result: works! (I think, smth is strange still- )
    Tried with cal map but with conf_thresh 0.5, result: works!

    Notes:
    1. Note that validation on the 10k (size of val for BDD) on one class (car) takes one hour (!) or 15s per iter.
    2. Note that validation on the 30% of the 10k (size of val for BDD) on the 4 car classes (car, truck, bus, train) takes 1 hours (!) or 30s per iter.

    # with mAP:
    nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python train.py --conf_thres 0.5 --amp "True" --log_path ./logs/onlybdd10k_FT_v0_bs_16_repeat_more_classes_with_mAP -p bdd10k -c 3 -b 16  -w weights/hybridnets_original_pretrained.pth --num_gpus 1 --optim adamw --lr 1e-6 --num_epochs 50' 2>&1 | tee -a onlybdd10k_FT_v0_bs_16_repeat_more_classes_with_mAP.txt & 
    
    # no mAP:
    nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python train.py --cal_map "False" --amp "True" --log_path ./logs/onlybdd10k_FT_v0_bs_16_repeat_more_classes -p bdd10k -c 3 -b 16  -w weights/hybridnets_original_pretrained.pth --num_gpus 1 --optim adamw --lr 1e-6 --num_epochs 50' 2>&1 | tee -a onlybdd10k_FT_v0_bs_16_repeat_more_classes.txt & 
    
    # with MUNIT output, no mAP:
    nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python train.py --munit_path /mnt/raid/home/eyal_michaeli/git/imaginaire/logs/2023_0421_1405_28_ampO1_lower_LR/inference_cp_400k_style_std_1.5_on_new_10k/ --cal_map "False" --amp "True" --log_path ./logs/onlybdd10k_FT_v0_bs_16_with_MUNIT_5_outputs -p bdd10k -c 3 -b 16  -w weights/hybridnets_original_pretrained.pth --num_gpus 1 --optim adamw --lr 1e-6 --num_epochs 50' 2>&1 | tee -a onlybdd10k_FT_v0_bs_16_with_MUNIT_5_outputs.txt & 

    # with MUNIT output, with mAP:
    nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python train.py --conf_thres 0.5 --munit_path /mnt/raid/home/eyal_michaeli/git/imaginaire/logs/2023_0421_1405_28_ampO1_lower_LR/inference_cp_400k_style_std_1.5_on_new_10k/ --cal_map "False" --amp "True" --log_path ./logs/onlybdd10k_FT_v0_bs_16_with_MUNIT_5_outputs -p bdd10k -c 3 -b 16  -w weights/hybridnets_original_pretrained.pth --num_gpus 1 --optim adamw --lr 1e-6 --num_epochs 50' 2>&1 | tee -a onlybdd10k_FT_v0_bs_16_with_MUNIT_5_outputs.txt & 
    
    # tensorboard:
    tensorboard --logdir=logs --port=6008

    """
    # print pid
    print('PID: ', os.getpid())
    opt = get_args()
    train(opt)

