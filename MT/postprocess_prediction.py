'''
python postprocess_prediction \
    --raw_pred eval/$EVAL_NAME/test_b5.rawpred \
    --sent_ids eval/$EVAL_NAME/test.ids \
    --final_pred eval/$EVAL_NAME/test_b5.pred \
'''

import argparse
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_pred", type=str)
    parser.add_argument("--sent_ids", type=str, default=None)
    parser.add_argument("--final_pred", type=str)
    parser.add_argument("--gt_ids", type=str, default=None)
    parser.add_argument("--sent_ids_prefix", type=str, default=None)

    args = parser.parse_args()
    return args

def final_postprocess(sent):
    for character in list(".,?!。，、…？！「」 "):
        sent = sent.replace(character, '')
    return sent

if __name__ == '__main__':
    args = get_args()

    with open(args.raw_pred) as f:
        raw_pred = f.read().split("\n")

    with open(args.gt_ids) as f:
        gt_ids = f.read().split("\n")


    if args.sent_ids:
        with open(args.sent_ids) as f:
            sent_ids = f.read().split("\n")
        assert len(raw_pred) == len(sent_ids)


    final_dict = {gt_id: '' for gt_id in gt_ids}


    for sent_id, pred in zip(sent_ids, raw_pred):
        if sent_id:
            if args.sent_ids_prefix is not None:
                sent_id = args.sent_ids_prefix + f"{int(sent_id)+1:04d}"
            if sent_id in final_dict:
                final_dict[sent_id] += pred


    out_lines = []
    for gt_id in gt_ids:
        out_lines.append(final_postprocess(final_dict[gt_id])+'\n')

    with open(args.final_pred, 'w') as f:
        f.writelines(out_lines)