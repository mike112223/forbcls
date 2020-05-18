
workdir = 'workdir'


img_norm_cfg = dict(mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
data = dict(
    val=dict(
        dataset=dict(
            type='Folder',
            directory='data/test',
            class_mode='categorical',
        ),
        transforms=[
            dict(type='Resize', img_scale=(128, 128)),
            dict(
                type='RandomAffineTransform',
                rotate_range=30,
                shear_range=0.2,
                shift_range=0.2,
                zoom_range=0.2,
            ),
            dict(type='RandomChannelShift', shift_range=20),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=1,
            num_workers=1,
            shuffle=True,
            drop_last=True,
        ),
    )
)


model = dict(
    type='Classifier',
    backbone=dict(
        type='MobileNet',
        arch='mobilenet_v2',
        num_classes=3,
        pretrained_ignore_layer='classifier',
        pretrained=True,
    ),
)
