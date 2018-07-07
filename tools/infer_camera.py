# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import logging
import time

from caffe2.python import workspace
import detectron.utils.env as envu

envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='../configs/DensePose_ResNet101_FPN_s1x-e2e.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='../DensePoseData/DensePose_ResNet101_FPN_s1x-e2e.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/home/yym/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    # parser.add_argument(
    #     'im_or_folder', help='image or folder of images', default=None
    # )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def main_densepose(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    # if os.path.isdir(args.im_or_folder):
    #     im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    # else:
    #     im_list = [args.im_or_folder]
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        # img = frame.copy()
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, frame, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        img, IUV, INDS = vis_utils.vis_one_image_uv(
            # frame[:, :, ::-1],  # BGR -> RGB for visualization
            frame,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            show_class=True,
            show_box=True,
            dataset=dummy_coco_dataset
        )

        # ADD CODE HERE
        t = time.time()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        plt.axis('off')
        ax1 = fig.add_subplot(131)
        plt.contour(IUV[:, :, 1] / 256., 10, linewidths=1)
        plt.contour(IUV[:, :, 2] / 256., 10, linewidths=1)
        plt.contour(INDS, linewidths=4)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        im1 = ax1.imshow(frame)
        im2 = ax2.imshow(frame)
        im3 = ax3.imshow(frame)
        # plt.contour(IUV[:, :, 1] / 256., 10, linewidths=1)
        # plt.contour(IUV[:, :, 2] / 256., 10, linewidths=1)
        # plt.contour(INDS, linewidths=4)
        im1.set_data(img)
        im2.set_data(img)
        im3.set_data(IUV)
        fig.canvas.draw()
        frametoshow = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                    sep=str(''))
        frametoshow = frametoshow.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        # frametoshow = cv2.cvtColor(frametoshow, cv2.COLOR_RGB2BGR)
        logger.info('Draw time: {:.3f}s'.format(time.time() - t))
        cv2.imshow("cam", frametoshow)
        k = cv2.waitKey(300) & 0xFF
        if k == 27:
            break

    plt.ioff()
    cap.release()
    cv2.destroyAllWindows()


def main_detectron(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file('/home/yym/Soft/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml')
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg('/home/yym/Soft/detectron/tools/model_final.pkl')
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        # img = frame.copy()
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, frame, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        img = vis_utils.vis_one_image_opencv(
            frame,
            cls_boxes,
            cls_segms,
            cls_keyps,
            thresh=0.7,
            show_box=True,
            dataset=dummy_coco_dataset,
            show_class=True
        )
        cv2.imshow("cam", img)
        k = cv2.waitKey(120) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main_detectron(args)
