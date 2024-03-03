readable_name = 'NegLabel_CLIPb16'
model = dict(
    type='ScalableClassifier',
    t=1,  ## Note: t = 1 means temperature is set to 0.01. See mmcls/models/classifiers/multi_modal.py L.188. 
    ngroup=100,
    classifier=dict(
        type='CLIPScalableClassifier',
        arch='ViT-B/16',
        train_dataset='imagenet',
        wordnet_database='./txtfiles/',
        # wordnet_database='./txtfiles_gt/',  ## OOD labels as negative label. ref: https://openreview.net/forum?id=xUO1HXz4an&noteId=UYGGqnkaSp
        neg_topk=0.15, # percentage 15% -> 10000
        emb_batchsize=1000,
        prompt_idx_pos=85, #[0,80]
        prompt_idx_neg=85,
        dump_neg=False,
        load_dump_neg=False,
        pencentile=0.95,
    )
)
pipline =[
          dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='./data/val',
        data_ann='./data/meta/val_labeled.txt',
        pipeline=pipline,
        train_label=None,
    ),
    ood_data=[
        dict(
            name='iNaturalist',
            type='FolderDataset',
            path='./data/ood_data/iNaturalist/images',
            pipeline=pipline,
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            path='./data/ood_data/SUN/images',
            pipeline=pipline,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='./data/ood_data/Places/images',
            pipeline=pipline,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='./data/ood_data/Textures/dtd/images_collate',
            pipeline=pipline,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
work_dir = './results/'
