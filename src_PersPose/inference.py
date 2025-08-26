import numpy as np
import torch, os, glob, torchvision, cv2
from common.visualize import array_show
import common.bbox as mybbox
from datasets.transform import Transform
from PIL import Image
from common.smpl_wrapper import SMPL
import exifread
import piexif
from pillow_heif import register_heif_opener
register_heif_opener()

detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to('cuda')
detector.eval()


def detect_human(img_file, bbox_scale=1.2):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    # scale = np.sqrt(img.size[0] * img.size[1] / 1000 / 1000)
    # if scale > 2.3:
    #     img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
    # else:
    #     scale = 1
    scale = 1
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    img_tensor = transform(img)
    with torch.no_grad():
        predictions = detector([img_tensor.to('cuda')])

    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']
    if boxes.shape[0] == 0 or (labels == 1).sum() == 0:
        return None
    bbox_idx = ((labels == 1) * scores).argmax()
    xyxy = (scale*boxes[bbox_idx]).tolist()
    return mybbox.center_wh2xywh(mybbox.scale_center_wh(mybbox.xywh2center_wh(mybbox.xyxy2xywh(xyxy)), bbox_scale))


@torch.no_grad()
def inference(args, model):
    img_files = []
    for suffix in ['jpg', 'png', 'HEIC']:
        img_files += glob.glob(os.path.join(args.inference, f'*.{suffix}'))
        args.log(os.path.join(args.inference, f'*.{suffix}'))
    args.log(f'inference on {len(img_files)} imgs')
    model.eval()
    transform = Transform(args)
    img_files = sorted(img_files)
    smpl = SMPL(gender='neutral', batch_size=1, device='cpu', dtype=torch.float32, hybrik_joints=False,num_betas=11)
    save_dir = args.inference + '_res'
    os.makedirs(save_dir, exist_ok=True)
    for img_file in img_files:
        # if len(os.path.basename(img_file).split('.')[-2].split('_')) > 7:
        #     f, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h = [int(each) for each in
        #                                                  os.path.basename(img_file).split('.')[-2].split('_')[-7:]]
        focal_length_35mm, img_w, img_h = None, None, None
        if os.path.basename(img_file).split('.')[-2][-4:] == '_f35':
            focal_length_35mm = int(os.path.basename(img_file).split('.')[-2].split('_')[-2])
            img_w, img_h = Image.open(img_file).size
        else:
            if str(img_file).lower().endswith('heic'):
                tags = exifread.process_file(open(img_file, 'rb'))
                if 'EXIF FocalLengthIn35mmFilm' in tags.keys():
                    focal_length_35mm = float(f'%s' % tags['EXIF FocalLengthIn35mmFilm'])
                # print(tags['EXIF FocalLength'])
                img = Image.open(img_file)
                img_file = os.path.join(save_dir, f"{os.path.basename(img_file)}.jpg")
                img.save(img_file)
                img_w, img_h = img.size
            else:
                img = Image.open(img_file)
                if img._getexif() is not None:
                    focal_length_35mm = img._getexif().get(41989)
                if focal_length_35mm is None:
                    if img.info.get('exif') is not None:
                        focal_length_35mm = piexif.load(img.info.get('exif')).get(41989)
                        if focal_length_35mm is None:
                            if 'Exif' in piexif.load(img.info.get('exif')).keys():
                                focal_length_35mm = piexif.load(img.info.get('exif')).get('Exif').get(41989)
                img_w, img_h = img.size
        if focal_length_35mm is None:
            focal_length_35mm = 50  # when horizontal FOV = 40 degree
            args.log(f'Image is not paired with focal length. {img_file}')
        f = focal_length_35mm / 36 * img_w  # Width of 35 mm film: 36mm.
        cx, cy = img_w / 2, img_h / 2
        box = detect_human(img_file)
        if box is None:
            args.log(f'No human detected in image. {img_file}')
            continue
        bbox_x, bbox_y, bbox_w, bbox_h = box
        data = {
            'file_name': img_file,
            'bbox': [bbox_x,bbox_y,bbox_w,bbox_h],
            'dataset_name': '3dhp',  # just to transform/preprocess images
            'cam_intrinsics': np.array([[f,0,cx],[0,f,cy],[0,0,1]]),
            'bbox_mask': np.zeros(4, ),
            'mask': np.zeros((256, 256)),
            'dp_uv': np.zeros((6890, 2)),
            'coco_kp17': np.zeros((17, 3)),
            '3dhp_j17': np.zeros((17, 3)),
            'dp_valid': np.zeros((6890,), dtype=bool),
        }
        img, target = transform(data, train=False)
        inp = {'img': img[None].to('cuda')}
        labels = {'cam_intrinsics':torch.tensor(target['cam_intrinsics'][None]).float().to('cuda'),'dataset_name':['3dhp']}
        output = model(inp,labels)

        centered_img_file = os.path.join(save_dir, f"centered_{os.path.basename(img_file)}.png")
        array_show(img, centered_img_file)

        smpl.render(output['beta'],output['theta'],output['transl'],[centered_img_file],labels['cam_intrinsics'],
                    res_path=save_dir,prefix='',suffix="_res.jpg")
        smpl.render_obj(output['beta'], output['theta'], res_path=save_dir, prefix=os.path.basename(centered_img_file))

    args.log(f'Done {args.session_dir}')
