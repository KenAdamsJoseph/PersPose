import torch
import numpy as np


def clip_xyxy(xyxy, boundary_xyxy):
    return [min(max(xyxy[0], boundary_xyxy[0]), boundary_xyxy[2]),
            min(max(xyxy[1], boundary_xyxy[1]), boundary_xyxy[3]),
            max(min(xyxy[2], boundary_xyxy[2]), boundary_xyxy[0]),
            max(min(xyxy[3], boundary_xyxy[3]), boundary_xyxy[1])]


def xyxy2xywh(xyxy):
    """
    :param xyxy: list(4)  x_min,y_min,x_max,y_max
    :return: list(4)x_min,y_min,width,height
    """
    x_min, y_min, x_max, y_max = xyxy
    w = x_max - x_min
    h = y_max - y_min
    return [x_min, y_min, w, h]


def xywh2xyxy(xywh):
    """
    :param xywh: list(4) x_min,y_min,width,height
    :return: list(4) x_min,y_min,x_max,y_max
    """
    x, y, w, h = xywh
    x_max = x + w
    y_max = y + h
    return [x, y, x_max, y_max]


def xywh2center_wh(xywh):
    """
    :param xywh: list(4) x_min,y_min,width,height
    :return: list(4) x_center,y_center,width,height
    """
    x, y, w, h = xywh
    cx = x + w / 2
    cy = y + h / 2
    return [cx, cy, w, h]


def center_wh2xywh(center_wh):
    """
    :param center_wh: list(4) x_center,y_center,width,height
    :return: xywh: list(4) x_min,y_min,width,height
    """
    cx, cy, w, h = center_wh
    x = cx - w / 2
    y = cy - h / 2
    return [x, y, w, h]


def scale_center_wh(center_wh, scale):
    """
    :param center_wh: list(4) x_center,y_center,width,height
    :param scale:
    :return: list(4) scaled center_wh
    """
    cx, cy, w, h = center_wh
    w_scaled, h_scaled = w * scale, h * scale
    return [cx, cy, w_scaled, h_scaled]


def scale_xywh(xywh, scale):
    """
    :param xywh: list(4) x_min,y_min,width,height
    :return: list(4) scaled xywh
    """
    center_wh = xywh2center_wh(xywh)
    scaled_center_wh = scale_center_wh(center_wh, scale)
    return center_wh2xywh(scaled_center_wh)


def adjust_w_h_ratio(w, h, w_h_ratio):
    # 调整（增大）bbox尺寸以符合目标宽高比
    old_ratio = w / h
    if old_ratio < w_h_ratio:
        # 宽度较小，按照高度调整
        new_h = h
        new_w = new_h * w_h_ratio
    else:
        new_w = w
        new_h = new_w / w_h_ratio
    return new_w, new_h
