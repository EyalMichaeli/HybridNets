from matplotlib import pyplot as plt
import torch
import numpy as np
import argparse
from tqdm.autonotebook import tqdm
import os
import logging
import cv2
from pathlib import Path
import psutil
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from utils import smp_metrics
from utils.utils import ConfusionMatrix, postprocess, scale_coords, process_batch, ap_per_class, fitness, \
    save_checkpoint, DataLoaderX, BBoxTransform, ClipBoxes, boolean_string, Params
from backbone import HybridNetsBackbone
from hybridnets.dataset import BddDataset
from hybridnets.custom_dataset import CustomDataset
from torchvision import transforms
import torch.nn.functional as F
from hybridnets.model import ModelWithLoss
from utils.constants import *


# same one as in config
normalization_stats = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

AMOUNT_TO_RUN_VAL_ON = 10000

def denormalize(img):
    img = np.transpose(img, (1, 2, 0))
    img = img * normalization_stats["std"] + normalization_stats["mean"]
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (2, 0, 1))
    return img

@torch.no_grad()
def val(model, val_generator, params, opt, seg_mode, is_training, pred_output_dir, **kwargs):
    """added tb_writer to write to tensorboard. not in use ATM"""
    model.eval()

    optimizer = kwargs.get('optimizer', None)
    scaler = kwargs.get('scaler', None)
    writer: SummaryWriter = kwargs.get('writer', None)
    epoch = kwargs.get('epoch', 0)
    step = kwargs.get('step', 0)
    best_fitness = kwargs.get('best_fitness', 0)
    best_loss = kwargs.get('best_loss', 0)
    best_epoch = kwargs.get('best_epoch', 0)

    conf_mat_output_dir = Path(pred_output_dir).parent / f'confusion_matrices'
    precision_recall_output_dir = Path(pred_output_dir).parent / f'precision_recall_curves'
    os.makedirs(pred_output_dir, exist_ok=True)
    os.makedirs(conf_mat_output_dir, exist_ok=True)
    os.makedirs(precision_recall_output_dir, exist_ok=True)

    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    stats, ap, ap_class = [], [], []
    iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()
    names = {i: v for i, v in enumerate(params.obj_list)}
    nc = len(names)
    ncs = 1 if seg_mode == BINARY_MODE else len(params.seg_list) + 1
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    s_seg = ' ' * (15 + 11 * 8)
    s = ('%-15s' + '%-11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mAcc')
    for i in range(len(params.seg_list)):
            s_seg += '%-33s' % params.seg_list[i]
            s += ('%-11s' * 3) % ('mIoU', 'IoU', 'Acc')
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iou_ls = [[] for _ in range(ncs)]
    acc_ls = [[] for _ in range(ncs)]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    if opt.cal_map:
        logging.info('Calculating mAP... will take one hour or more...')
        
    val_loader = tqdm(val_generator, ascii=True)
    for iter, data in enumerate(val_loader):
        imgs = data['img']
        annot = data['annot']
        seg_annot = data['segmentation']
        filenames = data['filenames']
        shapes = data['shapes']

        if opt.num_gpus == 1:
            imgs = imgs.cuda()
            annot = annot.cuda()
            seg_annot = seg_annot.cuda()

        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                seg_annot,
                                                                                                obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        seg_loss = seg_loss.mean()

        if iter < 10:
            # prepare vars for visualization

            img = imgs[0].cpu().numpy()

            step = 0 if step is None else step

            labels = annot[0]
            labels = labels[labels[:, 4] != -1]

            out = postprocess(imgs.detach(),
                                torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regression.detach(),
                                classification.detach(),
                                regressBoxes, clipBoxes,
                                opt.conf_thres, opt.iou_thres)  # 0.5, 0.3
            
            ou = out[0]
            nl = len(labels)

            pred = np.column_stack([ou['rois'], ou['scores']])
            pred = np.column_stack([pred, ou['class_ids']])
            pred = torch.from_numpy(pred).cuda()
            # done with visualization vars

            ### visualize bb
            # img = cv2.imread('/mnt/raid/home/eyal_michaeli/datasets/bdd/bdd100k/val' + filenames[i],
            #                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_UNCHANGED)

            # this is what I added:
            # img = denormalize(img)
            # img = np.transpose(img, (1, 2, 0))
            # img = np.clip(img * 255, 0, 255).astype(np.uint8)
            # Image.fromarray(img).save(f"{pred_output_dir}/pred+label-step_{step}-{iter}.jpg")

            ### done visualize bb

            # ### Visualization of seg
            # seg_0 = segmentation[i]
            # # logging.info('bbb', seg_0.shape)
            # seg_0 = torch.argmax(seg_0, dim = 0)
            # # logging.info('before', seg_0.shape)
            # seg_0 = seg_0.cpu().numpy().transpose(1, 2, 0)
            # # logging.info(seg_0.shape)
            # anh = np.zeros((384,640,3)) 
            # anh[seg_0 == 0] = (255,0,0)
            # anh[seg_0 == 1] = (0,255,0)
            # anh[seg_0 == 2] = (0,0,255)
            # anh = np.uint8(anh)
            # cv2.imwrite(f"{pred_output_dir}/segmentation-step_{step}-{i}.jpg", anh)    
            # ### done visulize seg

        if opt.cal_map:
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regression.detach(),
                              classification.detach(),
                              regressBoxes, clipBoxes,
                              opt.conf_thres, opt.iou_thres)  # 0.5, 0.3

            for i in range(annot.size(0)):
                seen += 1
                labels = annot[i]
                labels = labels[labels[:, 4] != -1]

                ou = out[i]
                nl = len(labels)

                pred = np.column_stack([ou['rois'], ou['scores']])
                pred = np.column_stack([pred, ou['class_ids']])
                pred = torch.from_numpy(pred).cuda()

                target_class = labels[:, 4].tolist() if nl else []  # target class

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), target_class))
                    # logging.info("here")
                    continue

                if nl:
                    pred[:, :4] = scale_coords(imgs[i][1:], pred[:, :4], shapes[i][0], shapes[i][1])
                    labels = scale_coords(imgs[i][1:], labels, shapes[i][0], shapes[i][1])

                    correct = process_batch(pred, labels, iou_thresholds)
                    if opt.plots:
                        confusion_matrix.process_batch(pred, labels)
                else:
                    correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

                # logging.info(stats)

            if seg_mode == MULTICLASS_MODE:
                segmentation = segmentation.log_softmax(dim=1).exp()
                _, segmentation = torch.max(segmentation, 1)  # (bs, C, H, W) -> (bs, H, W)
            else:
                segmentation = F.logsigmoid(segmentation).exp()

            tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation, seg_annot, mode=seg_mode,
                                                                   threshold=0.5 if seg_mode != MULTICLASS_MODE else None,
                                                                   num_classes=ncs if seg_mode == MULTICLASS_MODE else None)
            iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
            #         logging.info(iou)
            acc = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')

            for i in range(ncs):
                iou_ls[i].append(iou.T[i].detach().cpu().numpy())
                acc_ls[i].append(acc.T[i].detach().cpu().numpy())

        loss = cls_loss + reg_loss + seg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue

        loss_classification_ls.append(cls_loss.item())
        loss_regression_ls.append(reg_loss.item())
        loss_segmentation_ls.append(seg_loss.item())

    cls_loss = np.mean(loss_classification_ls)
    reg_loss = np.mean(loss_regression_ls)
    seg_loss = np.mean(loss_segmentation_ls)
    loss = cls_loss + reg_loss + seg_loss

    logging.info(
        'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
            epoch+1, opt.num_epochs if is_training else 0, cls_loss, reg_loss, seg_loss, loss))
    if is_training:
        writer.add_scalars('Loss', {'val': loss}, step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
        writer.add_scalars('Segmentation_loss', {'val': seg_loss}, step)

        # write memory usage to tensorboard
        writer.add_scalars('GPU_max_memory', {'train': torch.cuda.max_memory_allocated() / 1024 ** 3}, step)
        writer.add_scalars('GPU_memory', {'train': torch.cuda.memory_allocated() / 1024 ** 3}, step)
        # same for RAM
        writer.add_scalars('RAM', {'train': psutil.virtual_memory().percent}, step)

    if opt.cal_map:
        for i in range(ncs):
            iou_ls[i] = np.concatenate(iou_ls[i])
            acc_ls[i] = np.concatenate(acc_ls[i])
        # logging.info(len(iou_ls[0]))
        iou_score = np.mean(iou_ls)
        # logging.info(iou_score)
        acc_score = np.mean(acc_ls)

        miou_ls = []
        for i in range(len(params.seg_list)):
            if seg_mode == BINARY_MODE:
                # typically this runs once with i == 0
                miou_ls.append(np.mean(iou_ls[i]))
            else:
                miou_ls.append(np.mean( (iou_ls[0] + iou_ls[i+1]) / 2))

        for i in range(ncs):
            iou_ls[i] = np.mean(iou_ls[i])
            acc_ls[i] = np.mean(acc_ls[i])

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # logging.info(stats[3])

        # Count detected boxes per class
        # boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=1)

        ap50 = None
        precision_recall_output_path = Path(precision_recall_output_dir) / f'precision_recall_curve_step_{step}.png'

        # Compute metrics
        if len(stats) and stats[0].any():
            p, r, f1, ap, ap_class = ap_per_class(*stats, plot=opt.plots, output_path=precision_recall_output_path, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        logging.info(s_seg)
        logging.info(s)
        pf = ('%-15s' + '%-11i' * 2 + '%-11.3g' * 6) % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, acc_score)
        for i in range(len(params.seg_list)):
            tmp = i+1 if seg_mode != BINARY_MODE else i
            pf += ('%-11.3g' * 3) % (miou_ls[i], iou_ls[tmp], acc_ls[tmp])
        
        logging.info(pf)
        

        # Print results per class
        if opt.verbose and nc > 1 and len(stats):
            pf = '%-15s' + '%-11i' * 2 + '%-11.3g' * 4
            for i, c in enumerate(ap_class):
                logging.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        conf_matrix_output_path = Path(conf_mat_output_dir) / f'confusion_matrix_step_{step}.png'
        # Plots
        if opt.plots:
            confusion_matrix.plot(output_path=conf_matrix_output_path, names=list(names.values()))
            confusion_matrix.tp_fp()

        results = (mp, mr, map50, map, iou_score, acc_score, loss)
        fi = fitness(
            np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95, iou, acc, loss ]

        # if calculating map, save by best fitness
        if is_training and fi > best_fitness:
            best_fitness = fi
            ckpt = {'epoch': epoch,
                    'step': step,
                    'best_fitness': best_fitness,
                    'model': model.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()}
            logging.info(f"Saving checkpoint with best fitness: {fi[0]}")
            save_checkpoint(ckpt, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_best.pth')

        if is_training:
            # write memory usage to tensorboard
            writer.add_scalars('GPU_max_memory', {'train': torch.cuda.max_memory_allocated() / 1024 ** 3}, step+1)
            writer.add_scalars('GPU_memory', {'train': torch.cuda.memory_allocated() / 1024 ** 3}, step+1)
            # same for RAM
            writer.add_scalars('RAM', {'train': psutil.virtual_memory().percent}, step+1)

    else:
        # if not calculating map, save by best loss
        if is_training and loss + opt.es_min_delta < best_loss:
            best_loss = loss
            best_epoch = epoch

            save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_best.pth')

    # Early stopping
    if is_training and epoch - best_epoch > opt.es_patience > 0:
        logging.info('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
        exit(0)

    model.train()
    return (best_fitness, best_loss, best_epoch) if is_training else 0


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 python val.py --cal_map "True" --conf_thres 0.5 -p bdd10k -c 3 \
        -w logs/2023_0512_1354_02_bdd10k_repeat_base_one_class/checkpoints/hybridnets-d3_49_31250_best.pth \
            --pred_output_dir logs/logs/2023_0512_1354_02_bdd10k_repeat_base_one_class/
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    ap.add_argument('-bb', '--backbone', type=str,
                   help='Use timm to create another backbone replacing efficientnet. '
                   'https://github.com/rwightman/pytorch-image-models')
    ap.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficients of efficientnet backbone')
    ap.add_argument('-w', '--weights', type=str, default='weights/hybridnets.pth', help='/path/to/weights')
    ap.add_argument('-n', '--num_workers', type=int, default=4, help='Num_workers of dataloader')
    ap.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    ap.add_argument('-v', '--verbose', type=boolean_string, default=True,
                    help='Whether to print results per class when valing')
    ap.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    ap.add_argument('--plots', type=boolean_string, default=True,
                    help='Whether to plot confusion matrix when valing')
    ap.add_argument('--num_gpus', type=int, default=1,
                    help='Number of GPUs to be used (0 to use CPU)')
    ap.add_argument('--conf_thres', type=float, default=0.001,
                    help='Confidence threshold in NMS')
    ap.add_argument('--iou_thres', type=float, default=0.6,
                    help='IoU threshold in NMS')
    # add arg for pred_output_dir
    ap.add_argument('--pred_output_dir', type=str, default='preds', help='Directory to save predictions')
    args = ap.parse_args()

    compound_coef = args.compound_coef
    project_name = args.project
    weights_path = f'weights/hybridnets-d{compound_coef}.pth' if args.weights is None else args.weights

    params = Params(f'projects/{project_name}.yml')
    obj_list = params.obj_list
    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

    logging.info(f"Using {AMOUNT_TO_RUN_VAL_ON} images for validation")
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
        amount_to_run_on=AMOUNT_TO_RUN_VAL_ON
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(params.obj_list),
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=args.backbone,
                               seg_mode=seg_mode)
    
    try:
        model.load_state_dict(torch.load(weights_path))
    except:
        model.load_state_dict(torch.load(weights_path)['model'])
    model = ModelWithLoss(model, debug=False)
    model.requires_grad_(False)

    if args.num_gpus > 0:
        model.cuda()

    pred_output_dir = args.pred_output_dir
    val(model, val_generator, params, args, seg_mode, is_training=False, pred_output_dir=pred_output_dir)
