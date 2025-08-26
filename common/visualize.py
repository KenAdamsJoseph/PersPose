import numpy as np
import cv2
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt


# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
def array_show(array, save_path=None):
    if torch.is_tensor(array):
        array = array.detach().cpu()
    array_f = np.float64(array)
    if array.shape[0] == 3:
        array_f = array_f.transpose(1, 2, 0)
    array_f -= array_f.min()
    array_f /= array_f.max()
    img = Image.fromarray((array_f * 255).astype(np.uint8))
    if save_path is not None:
        img.save(save_path)
    else:
        img.show()


def draw_joints2D(joints2D, image_file='none', color='blue', radius=4, suffix='_j2d', image=None, out_f=None):
    """
    :param joints2D:
    :param image_file:
    :param color:
    :param radius:
    :param suffix:
    :param image:
    :return:
    """
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    if image is None:
        image = cv2.imread(image_file, 1)  # 默认值1->BGR  0->gray  -1->rgba
    else:
        image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR).copy()
        image -= image.min()
        image /= image.max()
        image = (image * 255).astype(np.uint8)
    for i in range(joints2D.shape[0]):
        pt1 = (int(joints2D[i, 0]), int(joints2D[i, 1]))
        cv2.circle(image, pt1, radius, colors[color][::-1].tolist(), -1)
    # cv2.imshow('title', image)
    prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    render_path = os.path.join(prj_root, 'render_res')
    os.makedirs(render_path, exist_ok=True)
    if out_f is None:
        out_f = os.path.join(render_path, f"{'_'.join(os.path.normpath(image_file).split(os.sep)[-2:])}{suffix}.jpg")
    cv2.imwrite(out_f, image)
    return image


if __name__ == '__main__':
    j2d = np.array([[550.90753, 366.18817],
                    [569.34607, 383.66238],
                    [540.7793, 385.95984],
                    [547.74994, 342.6577],
                    [570.93427, 461.26355],
                    [500.934, 456.15784],
                    [544.7382, 312.77377],
                    [554.4799, 537.893],
                    [459.22153, 526.6692],
                    [546.0533, 300.66727],
                    [552.29254, 553.9711],
                    [434.89838, 540.0509],
                    [554.93036, 251.08046],
                    [568.72546, 276.335],
                    [534.17834, 270.35345],
                    [567.8655, 238.72893],
                    [587.6763, 267.24896],
                    [520.4638, 254.16104],
                    [644.6479, 258.2986],
                    [471.566, 229.87973],
                    [688.95374, 247.61324],
                    [436.4673, 206.08586],
                    [701.7727, 247.19218],
                    [427.0285, 201.13065],
                    [580.6733, 211.03453],
                    [720.7055, 236.00279],
                    [411.14792, 192.34515],
                    [542.85284, 556.67126],
                    [423.45615, 542.4736]])
    img_file = r'./data/h36m/images/s_05_act_15_subact_02_ca_04/s_05_act_15_subact_02_ca_04_001356.jpg'
    draw_joints2D(j2d, img_file)
