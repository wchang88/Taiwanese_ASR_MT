import argparse
import numpy as np
import sacrebleu
from collections import defaultdict

COMET_MODEL = "Unbabel/wmt20-comet-da"
COMET_BATCH_SIZE = 64
BLEURT_BATCH_SIZE = 64

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("--comet-dir", type=str, default=None)
    parser.add_argument("--src", type=str)
    parser.add_argument("--calculate_cer", action='store_true')
    parser.add_argument("--trg_lang", type=str, default="")

    # experiments
    parser.add_argument("--sentlen", action="store_true",
        help="splitting sentences by their length, 0-5, 6-10, 11-15, 16-20, 21-30, 30+")
    parser.add_argument("--propn", action="store_true",
        help="splitting sentences by number of propns, 0, 1, 2+")
    parser.add_argument("--propn_norm", action="store_true",
        help="splitting sentences by (number of propns normalized by sent len), 0.05, 0.1, 0.15, 0.2, 0.2+")
    parser.add_argument("--verb", action="store_true",
        help="splitting sentences by number of verbs, 0, 1, 2, 3, 4, 5+")
    parser.add_argument("--verb_norm", action="store_true",
        help="splitting sentences by (number of verbs normalized by sent len), 0.05, 0.1, 0.15, 0.2, 0.2+")
    
    args = parser.parse_args()
    return args

def calculate_bleu(hyps, refs):
    # gets corpus-level non-ml evaluation metrics
    # corpus-level BLEU
    kwargs = {'tokenize': 'zh'} if args.trg_lang=='zh' else {}
    bleu = sacrebleu.metrics.BLEU(**kwargs)
    score = bleu.corpus_score(hyps, [refs]).format()
    return score

def calculate_comet(hyps, refs, srcs, comet_model):
    comet_input = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
    ]
    # sentence-level and corpus-level COMET
    comet_sentscores, comet_score = comet_model.predict(
        comet_input, batch_size=COMET_BATCH_SIZE, sort_by_mtlen=True
    )

    return comet_score


def get_edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def calculate_cer(hyps, refs):
    cumul_cer = 0
    for hyp, ref in zip(hyps, refs):
        cumul_cer += get_edit_distance(hyp, ref) / len(ref)
    return cumul_cer / len(hyps)


def calculate_scores(hyps, refs, srcs, comet_model):
    bleu_score = calculate_bleu(hyps, refs)
    comet_score = calculate_comet(hyps, refs, srcs, comet_model)
    return bleu_score, comet_score

def getSentLen(sent):
    return len(sent.split())

def split_data_by_len(hyps, refs, srcs):
    hyps_dict, refs_dict, srcs_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for src, hyp, ref in zip(srcs, hyps, refs):
        src_sent_len = getSentLen(src)
        if src_sent_len <= 5:
            hyps_dict["5"].append(hyp)
            srcs_dict["5"].append(src)
            refs_dict["5"].append(ref)
        elif src_sent_len <=10:
            hyps_dict["10"].append(hyp)
            srcs_dict["10"].append(src)
            refs_dict["10"].append(ref)
        elif src_sent_len <=15:
            hyps_dict["15"].append(hyp)
            srcs_dict["15"].append(src)
            refs_dict["15"].append(ref)
        elif src_sent_len <=20:
            hyps_dict["20"].append(hyp)
            srcs_dict["20"].append(src)
            refs_dict["20"].append(ref)
        elif src_sent_len <=30:
            hyps_dict["30"].append(hyp)
            srcs_dict["30"].append(src)
            refs_dict["30"].append(ref)
        else:
            hyps_dict["30+"].append(hyp)
            srcs_dict["30+"].append(src)
            refs_dict["30+"].append(ref)

    return hyps_dict, refs_dict, srcs_dict

def get_propn_cnt(sent):
    propn_cnt = 0
    doc = nlp(sent)
    for token in doc:
        if token.pos_ == "PROPN":
            propn_cnt += 1
    return propn_cnt, len(doc)

def get_verb_cnt(sent):
    verb_cnt = 0
    doc = nlp(sent)
    for token in doc:
        if token.pos_ == "VERB":
            verb_cnt += 1
    return verb_cnt, len(doc)

def split_data_by_propn_cnt(hyps, refs, srcs):
    hyps_dict, refs_dict, srcs_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for src, hyp, ref in zip(srcs, hyps, refs):
        propn_cnt, sent_len = get_propn_cnt(ref)
        if propn_cnt == 0:
            hyps_dict["0"].append(hyp)
            srcs_dict["0"].append(src)
            refs_dict["0"].append(ref)
        elif propn_cnt == 1:
            hyps_dict["1"].append(hyp)
            srcs_dict["1"].append(src)
            refs_dict["1"].append(ref)
        else:
            hyps_dict["2+"].append(hyp)
            srcs_dict["2+"].append(src)
            refs_dict["2+"].append(ref)

    return hyps_dict, refs_dict, srcs_dict

def split_data_by_propn_norm_cnt(hyps, refs, srcs):
    hyps_dict, refs_dict, srcs_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for src, hyp, ref in zip(srcs, hyps, refs):
        propn_cnt, sent_len = get_propn_cnt(ref)
        propn_cnt /= sent_len
        if propn_cnt <= 0.05:
            hyps_dict["0.05"].append(hyp)
            srcs_dict["0.05"].append(src)
            refs_dict["0.05"].append(ref)
        elif propn_cnt <= 0.1:
            hyps_dict["0.1"].append(hyp)
            srcs_dict["0.1"].append(src)
            refs_dict["0.1"].append(ref)
        elif propn_cnt <= 0.15:
            hyps_dict["0.15"].append(hyp)
            srcs_dict["0.15"].append(src)
            refs_dict["0.15"].append(ref)
        elif propn_cnt <= 0.2:
            hyps_dict["0.2"].append(hyp)
            srcs_dict["0.2"].append(src)
            refs_dict["0.2"].append(ref)
        else:
            hyps_dict["0.2+"].append(hyp)
            srcs_dict["0.2+"].append(src)
            refs_dict["0.2+"].append(ref)

    return hyps_dict, refs_dict, srcs_dict

def split_data_by_verb_cnt(hyps, refs, srcs):
    hyps_dict, refs_dict, srcs_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for src, hyp, ref in zip(srcs, hyps, refs):
        verb_cnt, sent_len = get_verb_cnt(ref)
        if verb_cnt == 0:
            hyps_dict["0"].append(hyp)
            srcs_dict["0"].append(src)
            refs_dict["0"].append(ref)
        elif verb_cnt == 1:
            hyps_dict["1"].append(hyp)
            srcs_dict["1"].append(src)
            refs_dict["1"].append(ref)
        elif verb_cnt == 2:
            hyps_dict["2"].append(hyp)
            srcs_dict["2"].append(src)
            refs_dict["2"].append(ref)
        elif verb_cnt == 3:
            hyps_dict["3"].append(hyp)
            srcs_dict["3"].append(src)
            refs_dict["3"].append(ref)
        elif verb_cnt == 4:
            hyps_dict["4"].append(hyp)
            srcs_dict["4"].append(src)
            refs_dict["4"].append(ref)
        else:
            hyps_dict["5+"].append(hyp)
            srcs_dict["5+"].append(src)
            refs_dict["5+"].append(ref)

    return hyps_dict, refs_dict, srcs_dict

def split_data_by_verb_norm_cnt(hyps, refs, srcs):
    hyps_dict, refs_dict, srcs_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for src, hyp, ref in zip(srcs, hyps, refs):
        verb_cnt, sent_len = get_verb_cnt(ref)
        normalized_val = verb_cnt / sent_len
        if normalized_val <= 0.05:
            hyps_dict["0.05"].append(hyp)
            srcs_dict["0.05"].append(src)
            refs_dict["0.05"].append(ref)
        elif normalized_val <= 0.1:
            hyps_dict["0.1"].append(hyp)
            srcs_dict["0.1"].append(src)
            refs_dict["0.1"].append(ref)
        elif normalized_val <= 0.15:
            hyps_dict["0.15"].append(hyp)
            srcs_dict["0.15"].append(src)
            refs_dict["0.15"].append(ref)
        elif normalized_val <= 0.2:
            hyps_dict["0.2"].append(hyp)
            srcs_dict["0.2"].append(src)
            refs_dict["0.2"].append(ref)
        else:
            hyps_dict["0.2+"].append(hyp)
            srcs_dict["0.2+"].append(src)
            refs_dict["0.2+"].append(ref)

    return hyps_dict, refs_dict, srcs_dict

def main(args):
    with open(args.hyp, encoding="utf-8") as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]
    with open(args.ref, encoding="utf-8") as ref_f:
        refs = [line.strip() for line in ref_f.readlines()]

    if args.comet_dir is not None:
        from comet import download_model, load_from_checkpoint

        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src) as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMET_MODEL, args.comet_dir)
        comet_model = load_from_checkpoint(comet_path)

        if args.sentlen:
            hyps_dict, refs_dict, srcs_dict = split_data_by_len(hyps, refs, srcs)
            for key in sentlen_bins:
                hyps, refs, srcs = hyps_dict[key], refs_dict[key], srcs_dict[key]
                print(f"Sentence length:{key}")
                print(f"Number of sentences:{len(srcs)}")
                bleu_score, comet_score = calculate_scores(hyps, refs, srcs, comet_model)
                print(bleu_score)
                print(f"COMET = {comet_score:.4f}\n")

        if args.propn:
            hyps_dict, refs_dict, srcs_dict = split_data_by_propn_cnt(hyps, refs, srcs)
            for key in propn_bins:
                hyps, refs, srcs = hyps_dict[key], refs_dict[key], srcs_dict[key]
                print(f"Number of PROPN in sentence:{key}")
                print(f"Number of sentences:{len(srcs)}")
                bleu_score, comet_score = calculate_scores(hyps, refs, srcs, comet_model)
                print(bleu_score)
                print(f"COMET = {comet_score:.4f}\n")
        
        if args.propn_norm:
            hyps_dict, refs_dict, srcs_dict = split_data_by_propn_norm_cnt(hyps, refs, srcs)
            for key in propn_norm_bins:
                hyps, refs, srcs = hyps_dict[key], refs_dict[key], srcs_dict[key]
                print(f"Number of PROPN(normalized by sent len) in sentence:{key}")
                print(f"Number of sentences:{len(srcs)}")
                bleu_score, comet_score = calculate_scores(hyps, refs, srcs, comet_model)
                print(bleu_score)
                print(f"COMET = {comet_score:.4f}\n")

        if args.verb:
            hyps_dict, refs_dict, srcs_dict = split_data_by_verb_cnt(hyps, refs, srcs)
            for key in verb_bins:
                hyps, refs, srcs = hyps_dict[key], refs_dict[key], srcs_dict[key]
                print(f"Number of VERB in sentence:{key}")
                print(f"Number of sentences:{len(srcs)}")
                bleu_score, comet_score = calculate_scores(hyps, refs, srcs, comet_model)
                print(bleu_score)
                print(f"COMET = {comet_score:.4f}\n")

        if args.verb_norm:
            hyps_dict, refs_dict, srcs_dict = split_data_by_verb_norm_cnt(hyps, refs, srcs)
            for key in verb_norm_bins:
                hyps, refs, srcs = hyps_dict[key], refs_dict[key], srcs_dict[key]
                print(f"Number of VERB(normalized by sent len) in sentence:{key}")
                print(f"Number of sentences:{len(srcs)}")
                bleu_score, comet_score = calculate_scores(hyps, refs, srcs, comet_model)
                print(bleu_score)
                print(f"COMET = {comet_score:.4f}\n")

        if not args.sentlen and not args.propn and not args.verb_norm:
            bleu_score, comet_score = calculate_scores(hyps, refs, srcs, comet_model)
            print(bleu_score)
            print(f"COMET = {comet_score:.4f}")
            if args.calculate_cer:
                cer = calculate_cer(hyps, refs)
                print(f"CER = {cer:.4f}")
        

if __name__ == "__main__":
    sentlen_bins = ["5","10","15","20","30","30+"]
    propn_bins = ["0", "1", "2+"]
    verb_bins = ["0", "1", "2", "3", "4", "5+"]
    propn_norm_bins = ["0.05", "0.1", "0.15", "0.2", "0.2+"]
    verb_norm_bins = ["0.05", "0.1", "0.15", "0.2", "0.2+"]
    
    args = get_args()

    if args.propn or args.propn_norm or args.verb or args.verb_norm:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
    main(args)
