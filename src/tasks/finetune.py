from argparse import Namespace

from vqa.vqa import VQA

import utils as u

args = Namespace(train='train,nominival',
                 valid='',
                 output='snap/vqa/finetune',
                 tiny=True,
                 test=None,
                 batch_size=16,
                 epochs=4,
                 optim='bert',

                 lr=5e-05,
                 dropout=0.1,
                 seed=9595,
                 fast=False,
                 tqdm=True,
                 load=None,
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

print('Splits in Train data:', vqa.train_tuple.dataset.splits)
if vqa.valid_tuple is not None:
    print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
    print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
else:
    print("DO NOT USE VALIDATION")
vqa.train(vqa.train_tuple, vqa.valid_tuple)
