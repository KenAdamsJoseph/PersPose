import sys
import os
import os.path as osp
import argparse
import logging
import yaml
import json
import wandb
import time
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from warmup_scheduler_pytorch import WarmUpScheduler
from contextlib import contextmanager
from glob import glob
from tqdm import tqdm

from metric import cal_loss_metric
from PersPose import PersPose
from inference import inference

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch.autograd.set_detect_anomaly(True)
# mp.set_sharing_strategy('file_system')
# os.environ["WANDB_MODE"] = "offline"

def get_model():
    model = PersPose(args)
    if args.ddp:
        args.log('Using SyncBatchNorm()')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # load pretrained backbone ckpt
    if args.pretrained_ckpt != '' and args.ckpt == '':
        args.log(f'loading pretrained model {args.pretrained_ckpt}')
        pretrained_ckpt_dict = torch.load(args.pretrained_ckpt, map_location='cpu')
        if 'model' in pretrained_ckpt_dict.keys():
            pretrained_ckpt_dict = pretrained_ckpt_dict['model']
        model.load_state_dict(pretrained_ckpt_dict, strict=True)

    return model


def get_dataset():
    from datasets.mixdataset import MixDataset
    from datasets.cocodp import CocoDp
    from datasets.humanmesh import HumanMesh
    from datasets.coco import COCO
    from datasets.hp3d import HP3D

    coco = COCO()
    hp3d = HP3D()
    hp3d_test = HP3D(split='test')

    h36m_train = HumanMesh('h36m', r"./data/h36m/densepose_train_5.pt", r"./data/h36m/images")
    h36m_test = HumanMesh('h36m', r"./data/h36m/densepose_test_5.pt", r"./data/h36m/images", 4)
    pw3d_train = HumanMesh('3dpw', r"./data/3dpw/densepose_train.pt", r"./data/3dpw")
    pw3d_test = HumanMesh('3dpw', r"./data/3dpw/densepose_test.pt", r"./data/3dpw")

    if args.mix_ratio[-1]+args.mix_ratio[-2]>0:
        bedlam_train = HumanMesh('bedlam', r"./data/bedlam/densepose_train_3.pt", r"./data/bedlam/data")
        bedlam_val = HumanMesh('bedlam', r"./data/bedlam/densepose_val_3.pt", r"./data/bedlam/data")
        train_set = MixDataset([
            h36m_train, coco, hp3d, pw3d_train, bedlam_train, bedlam_val
        ], args=args, train=True, ratio=args.mix_ratio, n=int(3e5))  # n % (args.batch_size * args.world_size) == 0
    else:
        train_set = MixDataset([
            h36m_train, coco, hp3d, pw3d_train,
        ], args=args, train=True, ratio=args.mix_ratio[:-2], n=int(3e5))  # n % (args.batch_size * args.world_size) == 0

    pw3d_test = MixDataset([pw3d_test, ], args=args, train=False)
    h36m_test = MixDataset([h36m_test, ], args=args, train=False)
    hp3d_test = MixDataset([hp3d_test, ], args=args, train=False)
    pw3d_test.epoch = 0
    h36m_test.epoch = 0
    hp3d_test.epoch = 0
    test_sets = [h36m_test, pw3d_test, hp3d_test]
    return train_set, test_sets


def get_dataloader():
    train_set, test_sets = get_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.ddp else None
    test_sampler = [torch.utils.data.distributed.DistributedSampler(each) if args.ddp else None for each in test_sets]
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=(not args.ddp),  # the sampler will shuffle
        sampler=train_sampler, persistent_workers=False)
    # persistent_workers will ignore the modification of dataset during traing, e.g., data_loader_train.dataset.epoch = epoch
    test_loaders = [torch.utils.data.DataLoader(
        dataset=test_sets[idx], batch_size=args.batch_size_eval, num_workers=args.num_workers,
        shuffle=(not args.ddp), sampler=test_sampler[idx], persistent_workers=False
    ) for idx in range(len(test_sets))]
    args.log('Dataloader prepared')
    if args.ddp: torch.distributed.barrier()
    return train_loader, test_loaders


def get_optimizer(model_):
    # the initial values of opt for all ddp workers should be the same   #achieved by the same random seed
    lr_base = args.lr
    no_decay_keywords = model_.no_weight_decay_keywords if hasattr(model_, 'no_weight_decay_keywords') else []
    lr_coe = []
    param_groups = []
    for name, param in model_.named_parameters():
        if not param.requires_grad:
            continue
        lr_coe_each = 1.0
        for module_name in args.lr_coefficient.keys():
            if name.startswith(module_name + '.'):
                lr_coe_each = args.lr_coefficient[module_name]
                break
        lr_coe.append(lr_coe_each)
        weight_decay_each = args.weight_decay
        if len(param.shape) == 1 or name.endswith(".bias"):
            weight_decay_each = 0.
        for nd in no_decay_keywords:
            if nd in name:
                weight_decay_each = 0.
                break
        param_groups.append({'params': param, 'lr': lr_base * lr_coe_each, 'weight_decay': weight_decay_each})
    args.lr_coe = lr_coe
    opt_ = torch.optim.AdamW(param_groups, lr=lr_base, weight_decay=args.weight_decay)
    return opt_


def get_lr():
    lr_base = args.lr
    if args.current_epoch < args.freeze:
        current_lr = lr_base
    elif args.current_epoch < args.freeze + args.warmup:
        current_lr = lr_base * (args.current_epoch - args.freeze + 1e-2) / args.warmup
    else:
        step_lr_epoch = args.current_epoch - args.freeze - args.warmup
        step_lr_ceo = args.scheduler_gamma ** (step_lr_epoch // args.scheduler_period)
        current_lr = lr_base * step_lr_ceo
    return current_lr


def lr_step(opt):
    current_lr = get_lr()
    for group_idx in range(len(opt.param_groups)):
        opt.param_groups[group_idx]['lr'] = args.lr_coe[group_idx] * current_lr


def create_logger(logdir, describe='demo', mute=False):
    os.makedirs(logdir, exist_ok=True)
    log_file = osp.join(logdir, f'log_{describe}.txt')
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file, format=head)
    logger_ = logging.getLogger()
    logger_.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)  # log信息输入stderr
    if not mute:
        logger_.addHandler(console)
    return logger_


@contextmanager
def torch_distributed_zero_first(rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    usage:
    with torch_distributed_zero_first(args.rank):
        do something
    """
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if rank == 0:
        torch.distributed.barrier()


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if args.reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # other reproducible factors: amp, version of {gpu, gpu driver, cuda, os, python pkg}
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def metric2str(metric_list):
    metric_info = '|'.join([f'{args.metric_label[i]} {metric_list[i]:.4f}' for i in range(len(args.metric_label))])
    return metric_info




@torch.no_grad()
def eval_model(model_, data_loader_, eval_label='', epoch=0, save_res=False, test_set_idx=0):
    model_.eval()
    all_metric = []
    eval_phar = tqdm(data_loader_, bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', leave=False)
    res_list = []
    for step, [inp, labels] in enumerate(eval_phar):
        if args.skim and step > 5: break
        args.current_step = step
        # label data is on cpu memory, but pred data is on gpu memory.
        # To save gpu memory, calculate loss on cpu, but cpu's slower then gpu.
        for k in inp.keys():
            if torch.is_tensor(inp[k]):
                if inp[k].dtype == torch.float64:
                    inp[k] = inp[k].float()
                inp[k] = inp[k].to('cuda')
        for k in labels.keys():
            if torch.is_tensor(labels[k]):
                if labels[k].dtype == torch.float64:
                    labels[k] = labels[k].float()
                labels[k] = labels[k].to('cuda')
        output = model_(inp, labels)
        loss, metric = cal_loss_metric(output, labels, inp, model_.module if args.ddp else model_)
        all_metric.append(metric)
        metric_info = metric2str(metric)
        eval_phar.set_description(f"🧲️> {eval_label}")  # |{metric_info}
        if save_res:
            for k in inp.keys():
                if torch.is_tensor(inp[k]):
                    inp[k] = inp[k].cpu().numpy()
            for k in labels.keys():
                if torch.is_tensor(labels[k]):
                    labels[k] = labels[k].cpu().numpy()
            for k in output.keys():
                if torch.is_tensor(output[k]):
                    output[k] = output[k].cpu().numpy()
            res_list.append({'labels':labels, 'output':output})  # 'inp':inp,
    if save_res:
        res_file = osp.join(args.session_dir, f'res_{args.rank}.torchsave')
        torch.save(res_list, res_file)
        args.log(f'****result saved****: {res_file}')
    if args.ddp: torch.distributed.barrier()
    metric_mean = torch.tensor(all_metric).mean(dim=0).tolist()
    metric_info = metric2str(metric_mean)
    elapsed = f"{tqdm.format_interval(eval_phar.format_dict['elapsed'])}"
    args.log(f"{eval_label}|{elapsed}|{metric_info}")
    eval_metric = metric_mean[args.metric4eval_idx[test_set_idx]]
    if args.ddp:  # compute average metric over all ddp workers
        # different workers may have different number of samples so the mean metric is not accurate!
        val_tensor = torch.tensor(metric_mean).cuda()
        dist.reduce(val_tensor, dst=0)
        if args.rank == 0:
            world_size = dist.get_world_size()
            metric_avg = val_tensor / world_size
            metric_info = metric2str(metric_avg)
            args.log(f'***ddp_reduce*** {eval_label}|{elapsed}|{metric_info}')
            eval_metric = metric_avg[args.metric4eval_idx[test_set_idx]]
            if args.wandb != '': wandb.log(
                {f'{args.metric_label[i]}_eval_{args.test_sets_name[test_set_idx]}': metric_avg[i] for i in
                 range(len(args.metric_label))},
                step=epoch)
        torch.distributed.barrier()
    else:
        if args.wandb != '': wandb.log(
            {f'{args.metric_label[i]}_eval_{args.test_sets_name[test_set_idx]}': metric_mean[i] for i in
             range(len(args.metric_label))},
            step=epoch)
    return eval_metric


def main():
    init_seeds(7)  # all ddp.workers use the same seed to produce the same initial param value
    model = get_model().to('cuda')
    opt = get_optimizer(model)
    args.start_epoch = 0
    scaler = GradScaler(enabled=not args.reproducible)
    if args.ckpt != '':
        ckpt_dict = torch.load(args.ckpt, map_location=f'cuda:{torch.cuda.current_device()}')
        model.load_state_dict(ckpt_dict['model'])
        args.start_epoch = ckpt_dict['epoch'] + 1
        # ckpt_dict['best_metric'] is not resumed
        opt.load_state_dict(ckpt_dict['optimizer'])
        if not args.reproducible and ckpt_dict['scaler'] is not None:
            scaler.load_state_dict(ckpt_dict['scaler'])
        args.log(f'ckpt loaded {args.ckpt}')
    if args.inference != '':
        inference(args, model)
        exit(0)
    if args.start_epoch < args.freeze:
        args.log('Freeze some layer')
        model.freeze(True)  # if get_optimizer after this line, frozen parameters will be ignored.
    if args.ddp:
        torch.distributed.barrier()
        model = DDP(model)  # , find_unused_parameters=True)
        # assure different ddp workers have the same param
        for param in model.parameters():
            # if all ddp.worker load the same ckpt, this may be redundant
            dist.broadcast(param.data, src=0)
        args.log(f'current device:{torch.cuda.current_device()}')
    data_loader_train, data_loaders_test = get_dataloader()
    best_metric = [1e10 for test_set_idx in range(len(args.test_sets_name))]
    if args.test:
        for test_set_idx in range(len(args.test_sets_name)):
            eval_label = f"Testing {args.test_sets_name[test_set_idx]:7s} rank:{args.rank}"
            _eval_metric = eval_model(model, data_loaders_test[test_set_idx], eval_label, save_res=False)
        exit(0)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        if epoch == args.freeze:
            args.log('Unfreeze some layer')
            if not args.ddp:
                model.freeze(False)
            else:
                model = model.module
                model.freeze(False)
                model = DDP(model)
        args.current_epoch = float(epoch)
        data_loader_train.dataset.epoch = epoch
        if args.ddp: data_loader_train.sampler.set_epoch(epoch)
        if not args.ddp or args.rank == 0:
            if args.wandb != '':
                wandb.log({f'lr': get_lr()}, step=epoch)
        all_metric = []
        train_pbar = tqdm(data_loader_train, bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', leave=False)
        train_label = f"Train|Epoch{str(epoch).zfill(3)}/{str(args.epochs).zfill(3)}|rank{args.rank}"
        grad_clip_n = 0
        for step, [inp, labels] in enumerate(train_pbar):
            if args.skim and step > 5: break
            args.current_step = step
            args.current_epoch = epoch + step / len(train_pbar)
            lr_step(opt)
            for k in inp.keys():
                if torch.is_tensor(inp[k]):
                    if inp[k].dtype == torch.float64:
                        inp[k] = inp[k].float()
                    inp[k] = inp[k].to('cuda')
            for k in labels.keys():
                if torch.is_tensor(labels[k]):
                    labels[k] = labels[k].to('cuda')
                    if labels[k].dtype == torch.float64:
                        labels[k] = labels[k].float()
            opt.zero_grad()
            with autocast(enabled=not args.reproducible):  # run with AMP
                output = model(inp, labels)
                loss, metric = cal_loss_metric(output, labels, inp, model.module if args.ddp else model)
            loss_sum = sum([args.loss_weight[i] * loss[i] for i in range(len(loss))])
            scaler.scale(loss_sum).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.grad_clip_ts)
            if grad_norm > args.grad_clip_ts:
                grad_clip_n += 1
            scaler.step(opt)
            scaler.update()
            all_metric.append(metric)
            metric_info = metric2str(metric)
            train_pbar.set_description(
                f"🧲️> {train_label}|lr {get_lr():.2e}|grad_norm {grad_norm:.2f}")  # |{metric_info}
        if args.ddp: torch.distributed.barrier()
        metric_mean = torch.tensor(all_metric).mean(dim=0).tolist()
        metric_info = metric2str(metric_mean)
        elapsed = f"{tqdm.format_interval(train_pbar.format_dict['elapsed'])}"
        args.log(f"{train_label}|{elapsed}|{metric_info}|lr {get_lr():.2e}|clip_grad {grad_clip_n}/{len(train_pbar)}")
        if args.ddp:  # compute average loss over all ddp workers
            val_tensor = torch.tensor(metric_mean).cuda()
            dist.reduce(val_tensor, dst=0)
            if args.rank == 0:
                world_size = dist.get_world_size()
                metric_avg = val_tensor / world_size  # average over all workers
                metric_info = metric2str(metric_avg)
                args.log(f'***ddp_reduce*** {train_label}|{elapsed}|{metric_info}')
                if args.wandb != '': wandb.log(
                    {f'{args.metric_label[i]}_train': metric_avg[i] for i in range(len(args.metric_label))}, step=epoch)
            torch.distributed.barrier()
        else:
            if args.wandb != '': wandb.log(
                {f'{args.metric_label[i]}_train': metric_mean[i] for i in range(len(args.metric_label))}, step=epoch)
        # save checkpoint
        ckpt_path = osp.join(args.session_dir, f"epoch{str(epoch).zfill(3)}.pth")
        if not args.ddp or args.rank == 0:
            torch.save({
                'model': model.module.state_dict() if args.ddp else model.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
                'optimizer': opt.state_dict(),
                'scaler': scaler.state_dict() if not args.reproducible else None,
            }, ckpt_path)
            for epoch_to_clear in range(0, epoch - args.ckpt_his_num):
                # remove too old ckpt
                file_path = osp.join(args.session_dir, f"epoch{str(epoch_to_clear).zfill(3)}.pth")
                if os.path.exists(file_path):
                    os.remove(file_path)
        # when there is something like saving ckpt that should only be done by rank 0, all ddp.workers wait.
        if args.ddp: torch.distributed.barrier()

        if args.validate:  # evaluate during training
            for test_set_idx in range(len(args.test_sets_name)):
                eval_label = f"Eval {args.test_sets_name[test_set_idx]:7s} |Epoch{str(epoch).zfill(3)}/{str(args.epochs).zfill(3)}|rank{args.rank}"
                eval_metric = eval_model(model, data_loaders_test[test_set_idx], eval_label,
                                         epoch=epoch, test_set_idx=test_set_idx)
                if not args.ddp or args.rank == 0:
                    try:  # create a file to record the metric value
                        open(f"{ckpt_path}.{args.test_sets_name[test_set_idx]}.{eval_metric:.4f}", 'w').close()
                    except Exception as e:
                        args.log(f"Failed to create metric file: {e}")
                if eval_metric < best_metric[test_set_idx]:
                    best_metric[test_set_idx] = eval_metric
                    if not args.ddp or args.rank == 0:
                        best_ckpt_file = osp.join(args.session_dir, f"best_{args.test_sets_name[test_set_idx]}.pth")
                        torch.save({
                            'model': model.module.state_dict() if args.ddp else model.state_dict(),
                            'epoch': epoch,
                            'best_metric': best_metric,
                            'optimizer': opt.state_dict(),
                            'scaler': scaler.state_dict() if not args.reproducible else None,
                        }, best_ckpt_file)
                        args.log(f'-------------best ckpt saved------------- '
                                 f'{args.metric_label[args.metric4eval_idx[test_set_idx]]}:{eval_metric:.4f} {best_ckpt_file}')
                        try:  # create a file to record the metric value
                            open(f"{best_ckpt_file}.{eval_metric:.4f}_epoch{str(epoch).zfill(3)}", 'w').close()
                        except Exception as e:
                            args.log(f"Failed to create metric file: {e}")
        if args.ddp: torch.distributed.barrier()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn') #when num_workers>0 cpu 100%
    default_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), './configs/config.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint path')
    parser.add_argument('--pretrained_ckpt', type=str, default='', help='checkpoint path')
    parser.add_argument('--auto_resume', action='store_true', help='auto-resume using the latest work_dir/*/latest.pth')

    parser.add_argument("--work_dir", type=str, default='workdir', help="experiment work path")
    parser.add_argument("--cfg", type=str, default=default_config_file, help="path of config.yaml")

    parser.add_argument('--test', action='store_true', help='only to test a given model')
    parser.add_argument('--inference', type=str, default='', help='image directory, infer on input imgs')
    parser.add_argument('--freeze', type=int, default=0, help='training epochs to freeze some module')
    parser.add_argument('--skim', action='store_true', help='skim over the progress')
    parser.add_argument('--wandb', type=str, default='', help='wandb_id https://wandb.ai/')

    args = parser.parse_args()
    args.start_time = time.strftime("%Y-%m%d-%H%M", time.localtime())
    args.prj_path = osp.dirname(osp.dirname(os.path.abspath(__file__)))
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1  # num of ddp workers
    args.rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1  # global rank of current ddp worker
    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ.keys() else -1
    args.current_epoch, args.current_step = 0.0, 0
    args.ddp = "LOCAL_RANK" in os.environ.keys()  # running in dist mode
    if args.ddp:
        torch.cuda.set_device(args.local_rank)  # set CUDA_VISIBLE_DEVICES to choose GPU
        dist.init_process_group(backend='nccl')

    # checkpoints under one workdir should be of the same model/shape
    args.work_dir = f'{args.work_dir}'  # _{args.joints_num}'

    # create logger
    session_list = sorted([each for each in glob(f"{args.work_dir}/*") if os.path.isdir(each)])
    last_session_file = f"{args.work_dir}/last_session.txt"
    if os.path.exists(last_session_file):
        session = int(np.loadtxt(last_session_file, dtype=np.int32)) + 1
    else:
        session = 1
    if len(session_list) != 0:
        session = max(session, int(os.path.basename(session_list[-1])) + 1)
    if args.ddp: torch.distributed.barrier()  # make assure all ddp workers have the same session_id
    if not args.ddp or args.rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        np.savetxt(last_session_file, [session], fmt='%d')
    args.session_dir = osp.join(args.work_dir, str(session).zfill(4))
    logger = create_logger(args.session_dir, describe=f'{args.start_time}_rank{args.rank}',
                           mute=args.ddp and args.rank != 0)
    # auto-resume
    if args.auto_resume:
        latest_ckpt = ''
        for session_dir in reversed(session_list):
            pth_ = sorted(glob(f"{session_dir}/epoch*.pth"))
            if len(pth_) > 0:
                latest_ckpt = pth_[-1]
                break
        if latest_ckpt != '':
            args.ckpt = latest_ckpt
            logging.info(f'auto-resume from the latest ckpt:{args.ckpt}')
        else:
            logging.info('cant find any ckpt')
    # load config.yaml and merge to args
    if args.cfg != '':
        with open(args.cfg, "r") as f:
            cfg_yaml = yaml.safe_load(f)
        for key, value in cfg_yaml.items():
            if hasattr(args, key):
                raise ValueError(f"Cfg key already exists {key}")
            else:
                setattr(args, key, value)
    args.metric4eval_idx = [args.metric_label.index(each) for each in args.metric4eval]
    logging.info(json.dumps(vars(args), indent=4))
    args.log = logger.info
    if args.wandb != '':
        args.wandb = f'{str(session).zfill(4)}_{args.wandb}'
        if not args.ddp or args.rank == 0:
            wandb.init(project=args.prj_name, config=args.__dict__, id=args.wandb)  # , resume="must")
    main()
