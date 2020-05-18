import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('../forbcls'))


def test_retina_forward():

    from forbcls.utils import Config
    from forbcls.model import build_model
    from forbcls.criterion import Criteria

    # init
    cfg_fp = os.path.join(os.path.abspath('config'), 'test.py')
    cfg = Config.fromfile(cfg_fp)

    model = build_model(cfg['model'])

    criterion = Criteria(
        cls_loss_cfg=cfg['criterion']['cls_loss'],
        reg_loss_cfg=cfg['criterion']['reg_loss'],
        num_classes=cfg['num_classes']
    )

    # input
    imgs = np.random.random((3, 3, 224, 224))
    imgs = torch.FloatTensor(imgs)

    # forward
    logits = model(imgs)
    print(logits)

    # reg_losses, cls_losses = criterion(preds_results, targets_results)
    # print(reg_losses, cls_losses)

    # cuda forward
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     imgs = imgs.cuda()
    #     # Test forward train
    #     gt_bboxes = [b.cuda() for b in mm_inputs['gt_bboxes']]
    #     gt_labels = [g.cuda() for g in mm_inputs['gt_labels']]

    #     preds_results, targets_results = model(
    #         imgs,
    #         img_metas,
    #         False,
    #         gt_bboxes=gt_bboxes,
    #         gt_labels=gt_labels,
    #     )

    #     reg_losses, cls_losses = criterion(preds_results, targets_results)
    #     print(reg_losses, cls_losses)


if __name__ == '__main__':
    test_retina_forward()
