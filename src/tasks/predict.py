import os
from argparse import Namespace

from vqa.vqa import VQA
from vqa.vqa import get_data_tuple

import utils as u

args = Namespace(train='train',
                 valid='',
                 test='test',
                 load='snap/vqa/finetune0/LAST',
                 output='snap/vqa/predict',
                 tiny=True,
                 batch_size=16,
                 epochs=4,
                 optim='bert',

                 lr=5e-05,
                 dropout=0.1,
                 seed=9595,
                 fast=False,
                 tqdm=True,
                 load_lxmert=None,
                 load_lxmert_qa='snap/pretrained/model',
                 from_scratch=False,
                 mce_loss=False,
                 llayers=9,
                 xlayers=5,
                 rlayers=5,
                 task_matched=False,
                 task_mask_lm=False,
                 task_obj_predict=False,
                 task_qa=False,
                 visual_losses='obj,attr,feat',
                 qa_sets=None,
                 word_mask_rate=0.15,
                 obj_mask_rate=0.15,
                 multiGPU=False,
                 num_workers=0)

u.handle_args(args)

vqa = VQA(args)

if args.load is not None:
    vqa.load(args.load)

vqa.predict(
    get_data_tuple(args.test, bs=950, args=args,
                   shuffle=False, drop_last=False),
    dump=os.path.join(args.output, 'test_predict.json')
)