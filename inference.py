"""
Code adapted from the detect.py script in the yolov5 repository.

"""

import argparse
import time
from pathlib import Path
import logging
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import PIL
import os

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np


LOGGER = logging.getLogger(__name__)

def frameIteratorFromVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        yield frame, video_path
    cap.release()

def frameIteratorFromFolder(folder_path):
    import glob
    for f in sorted(glob.glob(os.path.join(folder_path, "*"))):
        if f.endswith(".jpg") or f.endswith(".png"):
            yield cv2.imread(f), f
        elif f.endswith(".mp4"):
            yield from frameIteratorFromVideo(f)

def frameIteratorFromImage(image_path):
    yield cv2.imread(image_path), image_path

def frameIteratorFromStream(stream_path):
    cap = cv2.VideoCapture(stream_path)
    # drop frames in buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        yield frame, stream_path
    cap.release()

def getFrameIterator(data_path):
    if data_path.startswith("rtsp://"):
        return frameIteratorFromStream(data_path)
    if os.path.isdir(data_path):
        return frameIteratorFromFolder(data_path)
    elif os.path.isfile(data_path):
        if data_path.endswith(".mp4"):
            return frameIteratorFromVideo(data_path)
        else:
            return frameIteratorFromImage(data_path)
    else:
        raise ValueError("Data path must be a folder or a video file.")
    

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    save_img = not opt.nosave  # save inference images
    webcam = source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, sep="_"))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    if opt.save_src:
        (save_dir / 'src').mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference

    dataset = getFrameIterator(source)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    frame_count = 0
    for im0, path in dataset:
        loop_start = time.time()
        
        if opt.save_src:
            src_img = im0.copy()

        # resize and pad image
        
        img = letterbox(im0, new_shape=imgsz, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        wood_detected = False
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ""
            p = Path(path)  # to Path
            save_path = str(save_dir / p.name) + f'_{frame_count:010d}.jpg' # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + f'_{frame_count:010d}'  # img.txt

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        name = names[int(cls)]
                        label = f'{name} {conf:.2f}'
                        s += f"{label} "
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    
                        # HACK to store frames with wood for testing
                        if name == "wood":
                            wood_detected = True
                        

            # Stream results
            if view_img:
                viz = cv2.resize(im0, (960, 540))
                cv2.imshow(str(p), viz)
                k = cv2.waitKey(1)  # 1 millisecond
                if k == ord('q'):
                    break

            # Save results (image with detections)
            if save_img and wood_detected:
                cv2.imwrite(save_path, im0)
                if opt.save_src:
                    src_path = str(save_dir / "src" / p.name) + f'_{frame_count:010d}.jpg' # img.jpg
                    cv2.imwrite(src_path, src_img)

                LOGGER.info(f"Saved Frame to: {save_path}")


        # Print time (inference + NMS)
        LOGGER.info(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS. Total {1E3*(time.time() - loop_start):.1f}ms')


        frame_count += 1

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--save-src', action='store_true', help='Save source images (without detection bounding boxes)')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        detect()
