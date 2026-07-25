import copy
import logging
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.cuda import amp
import torch.distributed as dist

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


_BEST_MAP_EPS = 1e-3
_BEST_RANK1_EPS = 1e-6


def _should_eval_epoch(epoch, max_epoch=None):
    if max_epoch is not None and epoch == max_epoch:
        return True
    if epoch <= 10:
        return epoch % 2 == 0
    return epoch % 20 == 0


def _save_model(model, cfg, suffix):
    torch.save(
        _unwrap_model(model).state_dict(),
        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(suffix)),
    )


def _is_better_checkpoint(mAP, rank1, best_mAP, best_rank1):
    if mAP > best_mAP + _BEST_MAP_EPS:
        return True
    if abs(mAP - best_mAP) <= _BEST_MAP_EPS and rank1 > best_rank1 + _BEST_RANK1_EPS:
        return True
    return False


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


@torch.no_grad()
def _update_ema_model(student_model, teacher_model, momentum):
    student_state = _unwrap_model(student_model).state_dict()
    teacher_state = teacher_model.state_dict()
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(momentum).add_(student_value, alpha=1.0 - momentum)
        else:
            teacher_value.copy_(student_value)


def _feature_for_prototype(feat, feature_mode="global"):
    if isinstance(feat, (list, tuple)):
        if feature_mode == "global":
            return feat[0]
        if len(feat) == 5:
            return torch.cat([feat[0], feat[1] / 4, feat[2] / 4, feat[3] / 4, feat[4] / 4], dim=1)
        return feat[0]
    if feature_mode == "global" and feat.dim() == 2 and feat.size(1) > 768:
        return feat[:, :768]
    return feat


def _token_grid_shape(num_tokens, cfg):
    height, width = cfg.INPUT.SIZE_TRAIN
    stride_h, stride_w = (int(v) for v in cfg.MODEL.STRIDE_SIZE)
    grid_h = max(1, (height - int(cfg.INPUT.RB_MOL.MASK_AWARE_PATCH_SIZE)) // stride_h + 1)
    grid_w = max(1, (width - int(cfg.INPUT.RB_MOL.MASK_AWARE_PATCH_SIZE)) // stride_w + 1)
    if grid_h * grid_w != num_tokens:
        grid_h = max(1, int(round(num_tokens ** 0.5)))
        grid_w = max(1, int((num_tokens + grid_h - 1) // grid_h))
    return grid_h, grid_w


def _occlusion_mask_to_token_ratio(occ_mask, num_tokens, cfg):
    if occ_mask is None:
        return None
    if occ_mask.dim() == 3:
        occ_mask = occ_mask.unsqueeze(1)
    occ_mask = occ_mask.float()
    patch_size = int(cfg.INPUT.RB_MOL.MASK_AWARE_PATCH_SIZE)
    stride = tuple(int(v) for v in cfg.MODEL.STRIDE_SIZE)
    token_ratio = F.unfold(occ_mask, kernel_size=(patch_size, patch_size), stride=stride).mean(dim=1)
    if token_ratio.size(1) > num_tokens:
        token_ratio = token_ratio[:, :num_tokens]
    elif token_ratio.size(1) < num_tokens:
        token_ratio = F.pad(token_ratio, (0, num_tokens - token_ratio.size(1)), value=0.0)
    return token_ratio


def _region_balanced_reliability_select(reliability_score, clean_patches, keep_tokens, cfg):
    num_tokens = reliability_score.size(1)
    if num_tokens == 0:
        return None

    splits = sorted(max(0.0, min(1.0, float(value))) for value in cfg.INPUT.RB_MOL.REGION_BALANCED_SPLITS)
    regions = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1) if splits[i] < splits[i + 1]]
    if not regions:
        return None

    grid_h, grid_w = _token_grid_shape(num_tokens, cfg)
    token_indices = torch.arange(num_tokens, device=reliability_score.device)
    row_ids = torch.div(token_indices, grid_w, rounding_mode="floor").clamp(max=grid_h - 1)
    row_pos = (row_ids.float() + 0.5) / float(grid_h)

    base_quota = keep_tokens // len(regions)
    remainder = keep_tokens % len(regions)
    min_tokens = int(cfg.INPUT.RB_MOL.REGION_BALANCED_MIN_TOKENS)
    selected_scores = []
    selected_indices = []

    for region_idx, (start, end) in enumerate(regions):
        if region_idx == len(regions) - 1:
            region_mask = (row_pos >= start) & (row_pos <= end)
        else:
            region_mask = (row_pos >= start) & (row_pos < end)
        region_indices = token_indices[region_mask]
        if region_indices.numel() == 0:
            continue

        quota = base_quota + (1 if region_idx < remainder else 0)
        quota = min(int(region_indices.numel()), max(min_tokens, quota))
        if quota <= 0:
            continue

        region_scores = reliability_score.index_select(1, region_indices)
        top_scores, top_local_indices = torch.topk(region_scores, quota, dim=1, largest=True, sorted=False)
        selected_scores.append(top_scores)
        selected_indices.append(region_indices[top_local_indices])

    if not selected_scores:
        return None

    selected_scores = torch.cat(selected_scores, dim=1)
    selected_indices = torch.cat(selected_indices, dim=1)
    gather_indices = selected_indices.unsqueeze(2).expand(-1, -1, clean_patches.size(2))
    selected_patches = torch.gather(clean_patches, 1, gather_indices)
    return selected_scores, selected_patches


def _paired_reliable_teacher_anchor(clean_tokens, occ_tokens, occ_mask, cfg):
    clean_cls = clean_tokens[:, 0]
    clean_patches = clean_tokens[:, 1:]
    occ_patches = occ_tokens[:, 1:]
    if clean_patches.numel() == 0 or occ_patches.numel() == 0:
        return clean_cls

    num_tokens = min(clean_patches.size(1), occ_patches.size(1))
    clean_patches = clean_patches[:, :num_tokens]
    occ_patches = occ_patches[:, :num_tokens]

    clean_cls_norm = F.normalize(clean_cls, dim=1)
    clean_patch_norm = F.normalize(clean_patches, dim=2)
    occ_patch_norm = F.normalize(occ_patches, dim=2)

    identity_score = torch.sum(clean_patch_norm * clean_cls_norm.unsqueeze(1), dim=2)
    stability_score = torch.sum(clean_patch_norm * occ_patch_norm, dim=2)
    reliability_score = (
        float(cfg.INPUT.RB_MOL.RELIABILITY_ID_WEIGHT) * identity_score
        + float(cfg.INPUT.RB_MOL.RELIABILITY_STABILITY_WEIGHT) * stability_score
    )

    if cfg.INPUT.RB_MOL.MASK_AWARE_RELIABILITY:
        token_occ_ratio = _occlusion_mask_to_token_ratio(occ_mask, num_tokens, cfg)
        if token_occ_ratio is not None:
            token_occ_ratio = token_occ_ratio.to(reliability_score.device, dtype=reliability_score.dtype)
            reliability_score = reliability_score - float(cfg.INPUT.RB_MOL.MASK_AWARE_WEIGHT) * token_occ_ratio
            if cfg.INPUT.RB_MOL.MASK_AWARE_VISIBLE_ONLY:
                visible_mask = token_occ_ratio < float(cfg.INPUT.RB_MOL.MASK_AWARE_VISIBLE_THRESHOLD)
                has_visible = visible_mask.any(dim=1, keepdim=True)
                masked_score = reliability_score.masked_fill(~visible_mask, -1e4)
                reliability_score = torch.where(has_visible, masked_score, reliability_score)

    keep_tokens = int(round(num_tokens * float(cfg.INPUT.RB_MOL.RELIABILITY_TOPK_RATIO)))
    keep_tokens = max(int(cfg.INPUT.RB_MOL.RELIABILITY_TOPK_MIN_TOKENS), keep_tokens)
    keep_tokens = min(num_tokens, keep_tokens)

    selected = _region_balanced_reliability_select(reliability_score, clean_patches, keep_tokens, cfg)
    if selected is None:
        top_scores, top_indices = torch.topk(reliability_score, keep_tokens, dim=1, largest=True, sorted=False)
        gather_indices = top_indices.unsqueeze(2).expand(-1, -1, clean_patches.size(2))
        selected_scores = top_scores
        selected_patches = torch.gather(clean_patches, 1, gather_indices)
    else:
        selected_scores, selected_patches = selected

    weights = F.softmax(selected_scores / float(cfg.INPUT.RB_MOL.RELIABILITY_TEMP), dim=1)
    reliable_feat = torch.sum(selected_patches * weights.unsqueeze(2), dim=1)
    alpha = float(cfg.INPUT.RB_MOL.RELIABILITY_ALPHA)
    return (1.0 - alpha) * clean_cls + alpha * reliable_feat


@torch.no_grad()
def _teacher_anchor(teacher_model, clean_img, occ_img, occ_mask, camids, viewids, cfg):
    clean_tokens = teacher_model.forward_tokens(clean_img, cam_label=camids, view_label=viewids)
    occ_tokens = teacher_model.forward_tokens(occ_img, cam_label=camids, view_label=viewids)
    return _paired_reliable_teacher_anchor(clean_tokens, occ_tokens, occ_mask, cfg)


def _expand_prototype_memory(prototype_memory, prototype_valid, targets, feat_dim, device):
    required_size = int(targets.max().item()) + 1
    if prototype_memory is None:
        prototype_memory = torch.zeros(required_size, feat_dim, device=device, dtype=torch.float32)
        prototype_valid = torch.zeros(required_size, device=device, dtype=torch.bool)
    elif prototype_memory.size(0) < required_size:
        extra_size = required_size - prototype_memory.size(0)
        prototype_memory = torch.cat(
            [prototype_memory, torch.zeros(extra_size, feat_dim, device=device, dtype=prototype_memory.dtype)],
            dim=0,
        )
        prototype_valid = torch.cat(
            [prototype_valid, torch.zeros(extra_size, device=device, dtype=torch.bool)],
            dim=0,
        )
    return prototype_memory, prototype_valid


@torch.no_grad()
def _update_identity_prototypes(prototype_memory, prototype_valid, teacher_feat, targets, momentum):
    teacher_feat = F.normalize(teacher_feat.detach().float(), dim=1)
    prototype_memory, prototype_valid = _expand_prototype_memory(
        prototype_memory, prototype_valid, targets, teacher_feat.size(1), teacher_feat.device
    )
    for pid in targets.unique():
        pid_index = int(pid.item())
        pid_mask = targets == pid
        batch_proto = teacher_feat[pid_mask].mean(dim=0)
        if prototype_valid[pid_index]:
            prototype_memory[pid_index].mul_(momentum).add_(batch_proto, alpha=1.0 - momentum)
        else:
            prototype_memory[pid_index].copy_(batch_proto)
            prototype_valid[pid_index] = True
        prototype_memory[pid_index].copy_(F.normalize(prototype_memory[pid_index], dim=0))
    return prototype_memory, prototype_valid


def _prototype_occlusion_loss(student_feat, targets, prototype_memory, prototype_valid, temperature):
    student_feat = F.normalize(student_feat.float(), dim=1)
    valid_indices = torch.nonzero(prototype_valid, as_tuple=False).flatten()
    if valid_indices.numel() < 2:
        return student_feat.sum() * 0.0

    prototypes = F.normalize(prototype_memory[valid_indices].detach().float(), dim=1)
    logits = torch.matmul(student_feat, prototypes.t()) / float(temperature)
    target_positions = torch.full(
        (prototype_memory.size(0),), -1, device=targets.device, dtype=torch.long
    )
    target_positions[valid_indices] = torch.arange(valid_indices.numel(), device=targets.device)
    proto_targets = target_positions[targets]
    valid_samples = proto_targets >= 0
    if not valid_samples.any():
        return student_feat.sum() * 0.0
    return F.cross_entropy(logits[valid_samples], proto_targets[valid_samples])


def _sample_patch_size(height, width, area_range, aspect_range):
    image_area = height * width
    for _ in range(20):
        target_area = random.uniform(area_range[0], area_range[1]) * image_area
        aspect_ratio = random.uniform(aspect_range[0], aspect_range[1])
        patch_h = int(round((target_area * aspect_ratio) ** 0.5))
        patch_w = int(round((target_area / aspect_ratio) ** 0.5))
        if 1 <= patch_h < height and 1 <= patch_w < width:
            return patch_h, patch_w
    fallback_h = max(1, int(height * 0.35))
    fallback_w = max(1, int(width * 0.35))
    return min(fallback_h, height - 1), min(fallback_w, width - 1)


def _choose_patch_source(index, targets, avoid_same_id):
    batch_size = targets.size(0)
    candidates = list(range(batch_size))
    if avoid_same_id:
        current_pid = targets[index].item()
        candidates = [candidate for candidate in candidates if targets[candidate].item() != current_pid]
    candidates = [candidate for candidate in candidates if candidate != index]
    if not candidates:
        candidates = [candidate for candidate in range(batch_size) if candidate != index]
    if not candidates:
        return index
    return random.choice(candidates)


def _randint_from_range(min_value, max_value):
    min_value = int(round(min_value))
    max_value = int(round(max_value))
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    return random.randint(min_value, max_value)


def _sample_random_box(height, width, patch_h, patch_w):
    y = random.randint(0, height - patch_h)
    x = random.randint(0, width - patch_w)
    return y, x


def _sample_edge_box(height, width, patch_h, patch_w, edge_ratio):
    edge_h = max(1, int(round(height * float(edge_ratio))))
    edge_w = max(1, int(round(width * float(edge_ratio))))
    candidates = [
        (0, height - patch_h, 0, min(edge_w, width - patch_w)),
        (0, height - patch_h, max(0, width - edge_w - patch_w), width - patch_w),
        (0, min(edge_h, height - patch_h), 0, width - patch_w),
        (max(0, height - edge_h - patch_h), height - patch_h, 0, width - patch_w),
    ]
    y_min, y_max, x_min, x_max = random.choice(candidates)
    return _randint_from_range(y_min, y_max), _randint_from_range(x_min, x_max)


def _sample_body_box(height, width, patch_h, patch_w, body_x_range, body_y_range):
    x_min = float(body_x_range[0]) * width - patch_w / 2.0
    x_max = float(body_x_range[1]) * width - patch_w / 2.0
    y_min = float(body_y_range[0]) * height - patch_h / 2.0
    y_max = float(body_y_range[1]) * height - patch_h / 2.0
    x_min = max(0, min(width - patch_w, x_min))
    x_max = max(0, min(width - patch_w, x_max))
    y_min = max(0, min(height - patch_h, y_min))
    y_max = max(0, min(height - patch_h, y_max))
    return _randint_from_range(y_min, y_max), _randint_from_range(x_min, x_max)


def _sample_source_box(height, width, patch_h, patch_w, rbmol_cfg):
    if str(rbmol_cfg.SOURCE_MODE).lower() == "edge":
        return _sample_edge_box(height, width, patch_h, patch_w, rbmol_cfg.EDGE_RATIO)
    return _sample_random_box(height, width, patch_h, patch_w)


def _sample_target_box(height, width, patch_h, patch_w, rbmol_cfg):
    if str(rbmol_cfg.TARGET_MODE).lower() == "body":
        return _sample_body_box(height, width, patch_h, patch_w, rbmol_cfg.BODY_X_RANGE, rbmol_cfg.BODY_Y_RANGE)
    return _sample_random_box(height, width, patch_h, patch_w)


def make_rbmol_occluded_view(images, targets, cfg, return_occlusion_mask=False):
    rbmol_cfg = cfg.INPUT.RB_MOL
    occluded = images.clone()
    batch_size, _, height, width = images.shape
    occlusion_mask = images.new_zeros((batch_size, 1, height, width))

    for index in range(batch_size):
        if random.random() > float(rbmol_cfg.PROB):
            continue

        patch_h, patch_w = _sample_patch_size(height, width, rbmol_cfg.AREA_RANGE, rbmol_cfg.ASPECT_RANGE)
        source_index = _choose_patch_source(index, targets, rbmol_cfg.AVOID_SAME_ID)
        src_y, src_x = _sample_source_box(height, width, patch_h, patch_w, rbmol_cfg)
        dst_y, dst_x = _sample_target_box(height, width, patch_h, patch_w, rbmol_cfg)

        patch = images[source_index, :, src_y:src_y + patch_h, src_x:src_x + patch_w].clone()
        alpha = float(rbmol_cfg.BLEND_ALPHA)
        if alpha >= 1.0:
            occluded[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = patch
        else:
            target_patch = occluded[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w]
            occluded[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = (
                alpha * patch + (1.0 - alpha) * target_patch
            )
        occlusion_mask[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = 1.0

    if return_occlusion_mask:
        return occluded, occlusion_mask
    return occluded


def _evaluate(model, val_loader, evaluator, cfg, device):
    model.eval()
    evaluator.reset()
    for _, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))
    return evaluator.compute()


def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    occ_loss_meter = AverageMeter()
    proto_loss_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    best_mAP = 0.0
    best_rank1 = 0.0
    best_epoch = 0
    prototype_memory = None
    prototype_valid = None
    use_rbmol = cfg.INPUT.RB_MOL.ENABLED and cfg.INPUT.RB_MOL.CONSISTENCY_WEIGHT > 0

    teacher_model = None
    if use_rbmol:
        teacher_model = copy.deepcopy(_unwrap_model(model))
        teacher_model.to(device)
        teacher_model.eval()
        _set_requires_grad(teacher_model, False)

    if cfg.INPUT.RB_MOL.ENABLED:
        logger.info(
            "RB-MOL enabled: prob={}, area_range={}, aspect_range={}, source_mode={}, "
            "target_mode={}, occ_reid_weight={}, prototype_weight={}, prototype_temp={}, "
            "prototype_momentum={}, ema_momentum={}, reliability_temp={}, reliability_alpha={}, "
            "id_weight={}, stability_weight={}, topk_ratio={}, topk_min_tokens={}, "
            "region_splits={}, region_min_tokens={}, mask_weight={}, visible_threshold={}".format(
                cfg.INPUT.RB_MOL.PROB,
                cfg.INPUT.RB_MOL.AREA_RANGE,
                cfg.INPUT.RB_MOL.ASPECT_RANGE,
                cfg.INPUT.RB_MOL.SOURCE_MODE,
                cfg.INPUT.RB_MOL.TARGET_MODE,
                cfg.INPUT.RB_MOL.OCC_REID_WEIGHT,
                cfg.INPUT.RB_MOL.CONSISTENCY_WEIGHT,
                cfg.INPUT.RB_MOL.CONSISTENCY_TEMP,
                cfg.INPUT.RB_MOL.PROTOTYPE_MOMENTUM,
                cfg.INPUT.RB_MOL.EMA_MOMENTUM,
                cfg.INPUT.RB_MOL.RELIABILITY_TEMP,
                cfg.INPUT.RB_MOL.RELIABILITY_ALPHA,
                cfg.INPUT.RB_MOL.RELIABILITY_ID_WEIGHT,
                cfg.INPUT.RB_MOL.RELIABILITY_STABILITY_WEIGHT,
                cfg.INPUT.RB_MOL.RELIABILITY_TOPK_RATIO,
                cfg.INPUT.RB_MOL.RELIABILITY_TOPK_MIN_TOKENS,
                cfg.INPUT.RB_MOL.REGION_BALANCED_SPLITS,
                cfg.INPUT.RB_MOL.REGION_BALANCED_MIN_TOKENS,
                cfg.INPUT.RB_MOL.MASK_AWARE_WEIGHT,
                cfg.INPUT.RB_MOL.MASK_AWARE_VISIBLE_THRESHOLD,
            )
        )

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        occ_loss_meter.reset()
        proto_loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        if teacher_model is not None:
            teacher_model.eval()

        for n_iter, batch in enumerate(train_loader):
            if len(batch) == 5:
                img, vid, target_cam, target_view, _ = batch
            else:
                img, vid, target_cam, target_view = batch

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)
                occ_loss = None
                proto_loss = None

                if cfg.INPUT.RB_MOL.ENABLED:
                    img_occ, occ_mask = make_rbmol_occluded_view(
                        img, target, cfg, return_occlusion_mask=True
                    )
                    score_occ, feat_occ = model(img_occ, target, cam_label=target_cam, view_label=target_view)

                    if cfg.INPUT.RB_MOL.OCC_REID_WEIGHT > 0:
                        occ_loss = loss_fn(score_occ, feat_occ, target, target_cam)
                        loss = loss + float(cfg.INPUT.RB_MOL.OCC_REID_WEIGHT) * occ_loss

                    if use_rbmol and epoch > int(cfg.INPUT.RB_MOL.CONSISTENCY_WARMUP_EPOCHS):
                        teacher_anchor = _teacher_anchor(
                            teacher_model, img, img_occ, occ_mask, target_cam, target_view, cfg
                        )
                        teacher_anchor = _feature_for_prototype(
                            teacher_anchor, cfg.INPUT.RB_MOL.CONSISTENCY_FEATURE
                        )
                        prototype_memory, prototype_valid = _update_identity_prototypes(
                            prototype_memory,
                            prototype_valid,
                            teacher_anchor,
                            target,
                            float(cfg.INPUT.RB_MOL.PROTOTYPE_MOMENTUM),
                        )
                        student_anchor = _feature_for_prototype(
                            feat_occ, cfg.INPUT.RB_MOL.CONSISTENCY_FEATURE
                        )
                        proto_loss = _prototype_occlusion_loss(
                            student_anchor,
                            target,
                            prototype_memory,
                            prototype_valid,
                            float(cfg.INPUT.RB_MOL.CONSISTENCY_TEMP),
                        )
                        loss = loss + float(cfg.INPUT.RB_MOL.CONSISTENCY_WEIGHT) * proto_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if teacher_model is not None:
                _update_ema_model(model, teacher_model, float(cfg.INPUT.RB_MOL.EMA_MOMENTUM))

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            if occ_loss is not None:
                occ_loss_meter.update(occ_loss.item(), img.shape[0])
            if proto_loss is not None:
                proto_loss_meter.update(proto_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if cfg.INPUT.RB_MOL.ENABLED:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, OccLoss: {:.3f}, "
                        "ProtoLoss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                            epoch,
                            (n_iter + 1),
                            len(train_loader),
                            loss_meter.avg,
                            occ_loss_meter.avg,
                            proto_loss_meter.avg,
                            acc_meter.avg,
                            scheduler._get_lr(epoch)[0],
                        )
                    )
                else:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                            epoch,
                            (n_iter + 1),
                            len(train_loader),
                            loss_meter.avg,
                            acc_meter.avg,
                            scheduler._get_lr(epoch)[0],
                        )
                    )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                _save_model(model, cfg, epoch)

        if _should_eval_epoch(epoch, epochs):
            if cfg.MODEL.DIST_TRAIN and dist.get_rank() != 0:
                continue
            cmc, mAP, _, _, _, _, _ = _evaluate(model, val_loader, evaluator, cfg, device)
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            if _is_better_checkpoint(mAP, cmc[0], best_mAP, best_rank1):
                best_mAP = mAP
                best_rank1 = cmc[0]
                best_epoch = epoch
                _save_model(model, cfg, "best")
                logger.info(
                    "Best model updated - Epoch: {}, mAP: {:.1%}, Rank-1: {:.1%}".format(
                        best_epoch, best_mAP, best_rank1
                    )
                )
            logger.info(
                "Best Results - Epoch: {}, mAP: {:.1%}, Rank-1: {:.1%}".format(
                    best_epoch, best_mAP, best_rank1
                )
            )
            torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        model.to(device)

    cmc, mAP, _, _, _, _, _ = _evaluate(model, val_loader, evaluator, cfg, device)
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
