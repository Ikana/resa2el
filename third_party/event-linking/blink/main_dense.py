# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import sys
import os
import pickle

from tqdm import tqdm
import logging
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from colorama import init
from termcolor import colored

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder, to_bert_input
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.train_cross import modify, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
from event_linking_main import GetWikiID

HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding, weights_only=False)
        # print(type(candidate_encoding), candidate_encoding.size())
        # candidate_encoding = candidate_encoding[0:int(1174151/2)]
        # print(type(candidate_encoding), candidate_encoding.size())
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id


def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger):
    test_samples = []
    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            record = json.loads(line)
            record["label"] = str(record["label_id"])

            # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            if kb2id and len(kb2id) > 0:
                if record["label"] in kb2id:
                    record["label_id"] = kb2id[record["label"]]
                else:
                    continue

            # check that each entity id (label_id) is in the entity collection
            elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
                try:
                    key = int(record["label"].strip())
                    if key in wikipedia_id2local_id:
                        record["label_id"] = wikipedia_id2local_id[key]
                    else:
                        continue
                except:
                    continue

            # LOWERCASE EVERYTHING !
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
            test_samples.append(record)

    if logger:
        logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
    return test_samples


def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger
):
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger)
    return test_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None, device='cpu', cand_pool_path=None, id2title=None, save_topk_result=False):
    biencoder.model.eval()
    biencoder.to(device)
    labels = []
    nns = []
    all_scores = []

    nn_labels = []
    nn_context = []
    nn_candidates = []

    if save_topk_result:
        candidate_pool = torch.load(cand_pool_path, weights_only=False)
        id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
        original_id2title = pickle.load(id2title_file)

    if isinstance(candidate_encoding, np.ndarray):
        # Low-memory mode: encode all mentions, then stream the candidate
        # encoding through the GPU once (see blink/lowmem.py).
        assert not save_topk_result, "save_topk_result is not supported in low-memory mode"
        from blink import lowmem
        context_embs = []
        for batch in tqdm(dataloader):
            context_input, _, label_ids = batch
            with torch.no_grad():
                token_idx, segment_idx, mask = to_bert_input(context_input, biencoder.NULL_IDX)
                emb, _ = biencoder.model(
                    token_idx.to(device), segment_idx.to(device), mask.to(device), None, None, None
                )
            context_embs.append(emb.float().cpu())
            labels.extend(label_ids.data.numpy())
        with torch.no_grad():
            scores, indicies = lowmem.chunked_topk(
                torch.cat(context_embs), candidate_encoding, top_k, device=device
            )
        nns.extend(indicies)
        all_scores.extend(scores)
        return labels, nns, all_scores, None

    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding.to(device), device=device
                )
                scores, indicies = scores.topk(top_k)
                if device == 'cuda':
                    scores = scores.data.cpu().numpy()
                    indicies = indicies.data.cpu().numpy()
                else:
                    scores = scores.data.numpy()
                    indicies = indicies.data.numpy()
                if save_topk_result:
                    for i in range(context_input.size(0)):
                        inds = indicies[i]

                        pointer = -1
                        for j in range(top_k):
                            if id2title is not None:
                                title = id2title[inds[j].item()]
                                if title.replace(" ", "_") not in original_id2title[1]:
                                    continue
                                predict_id = GetWikiID(original_id2title, title)
                                if int(predict_id) == int(label_ids[i].item()):
                                    pointer = j
                                    break
                        if pointer == -1:
                            continue

                        # add examples in new_data
                        cur_candidates = candidate_pool[inds]
                        nn_context.append(context_input[i].cpu().tolist())
                        nn_candidates.append(cur_candidates.cpu().tolist())
                        nn_labels.append(pointer)

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
    }

    return labels, nns, all_scores, nn_data


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda", amp=None):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    # Precision of the crossencoder only (added for resa2el). The bi-encoder
    # keeps float32, so the candidate set does not change.
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    if amp == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        res = evaluate(crossencoder, dataloader, device, logger, context_len, zeshel=False, silent=False, amp_dtype=amp_dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits


def _modify_fast(context_input, candidate_input, max_seq_length):
    """Same result as train_cross.modify, with tensor ops instead of Python loops.

    For each mention, every candidate becomes the context followed by the
    candidate without its [CLS] token, cut to max_seq_length.
    """
    context_input = torch.as_tensor(context_input, dtype=torch.long)
    candidate_input = torch.as_tensor(candidate_input, dtype=torch.long)
    num_cand = candidate_input.size(1)
    ctx = context_input.unsqueeze(1).expand(-1, num_cand, -1)
    joined = torch.cat([ctx, candidate_input[:, :, 1:]], dim=2)
    return joined[:, :, :max_seq_length].contiguous()


def _ckpt_meta(ckpt_dir, meta):
    """Create ckpt_dir. Refuse to continue from a run with other settings."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != meta:
            raise ValueError("checkpoint {} is from a different run: {} != {}".format(ckpt_dir, old, meta))
    else:
        with open(path, "w") as f:
            json.dump(meta, f)


def _score_in_chunks(n, chunk, prep_fn, score_fn, ckpt_dir=None, prefetch=True, tag=""):
    """Score mentions [0, n) in chunks of `chunk`; return (index, scores) arrays.

    prep_fn(c0, c1) builds the crossencoder input of mentions c0..c1 (CPU).
    score_fn(prepared) scores it and returns (index, scores) as
    _run_crossencoder does. The result is the same as one call on all
    mentions, because each mention is scored on its own.
    Each finished chunk is saved in ckpt_dir (added for resa2el), so a
    stopped run continues where it was. With prefetch, the next chunk is
    prepared on the CPU while the GPU scores the current one. `tag` names
    the precision; a saved chunk with another tag is refused.
    """
    def part_path(c0, c1):
        return os.path.join(ckpt_dir, "cross_{:08d}_{:08d}.npz".format(c0, c1))

    bounds = [(c0, min(c0 + chunk, n)) for c0 in range(0, n, chunk)]
    todo = [b for b in bounds if not (ckpt_dir and os.path.exists(part_path(*b)))]
    todo_set = set(todo)  # decided once: the prefetch order follows this list
    pool = ThreadPoolExecutor(max_workers=1) if prefetch and todo else None
    pending = pool.submit(prep_fn, *todo[0]) if pool else None
    todo_idx = 0
    index_parts, score_parts = [], []
    try:
        for c0, c1 in bounds:
            t0 = time.time()
            if (c0, c1) not in todo_set:
                saved = np.load(part_path(c0, c1))
                part_index, part_scores = saved["index"], saved["scores"]
                assert part_index.shape[0] == c1 - c0, part_path(c0, c1)
                saved_tag = str(saved["tag"]) if "tag" in saved.files else ""
                if saved_tag != tag:
                    raise ValueError("{} was scored with {!r}, this run uses {!r}".format(part_path(c0, c1), saved_tag, tag))
                print("crossencoder: loaded {}-{} from {}".format(c0, c1, part_path(c0, c1)), flush=True)
            else:
                if pool:
                    assert todo[todo_idx] == (c0, c1), (todo[todo_idx], c0, c1)
                    prepared = pending.result()
                    todo_idx += 1
                    pending = pool.submit(prep_fn, *todo[todo_idx]) if todo_idx < len(todo) else None
                else:
                    prepared = prep_fn(c0, c1)
                t_wait = time.time() - t0
                part_index, part_scores = score_fn(prepared)
                part_index, part_scores = np.asarray(part_index), np.asarray(part_scores)
                del prepared
                if ckpt_dir:
                    tmp_path = part_path(c0, c1) + ".tmp.npz"
                    np.savez(tmp_path, index=part_index, scores=part_scores, tag=tag)
                    os.replace(tmp_path, part_path(c0, c1))
                if len(bounds) > 1:
                    print("crossencoder: {}/{} mentions, {:.0f}s (waited {:.0f}s for the input)".format(
                        c1, n, time.time() - t0, t_wait), flush=True)
            index_parts.append(part_index)
            score_parts.append(part_scores)
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
    return np.concatenate(index_parts), np.concatenate(score_parts)


def load_models(args, logger=None):

    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
            if getattr(args, "cross_batch", None):
                # Smaller batch so top-100 reranking fits a 24 GB GPU.
                crossencoder_params["eval_batch_size"] = args.cross_batch
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    if getattr(args, "lowmem_dir", None):
        # Low-memory mode: memory-mapped encoding, no entity catalogue
        # (the caller passes alternate_mapping, which replaces it).
        from blink import lowmem
        candidate_encoding = lowmem.load_encoding(args.lowmem_dir)
        return (
            biencoder,
            biencoder_params,
            crossencoder,
            crossencoder_params,
            candidate_encoding,
            {},
            {},
            {},
            {},
            None,
        )
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue,
        args.entity_encoding,
        faiss_index=getattr(args, 'faiss_index', None),
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer=None,
    test_data=None,
    device='cpu',
    alternate_mapping=None
):
    if not test_data and not args.test_mentions and not args.interactive:
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entitied (--test_entities)"
        )
        raise ValueError(msg)

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

    id2hyper = {}

    if alternate_mapping is not None:
        id2title = alternate_mapping[0]
        id2text = alternate_mapping[1]
        id2hyper = alternate_mapping[2]

    stopping_condition = False
    while not stopping_condition:

        samples = None

        if args.interactive:
            logger.info("interactive mode")

            # biencoder_params["eval_batch_size"] = 1

            # Load NER model
            import blink.ner as NER  # only interactive mode needs flair
            ner_model = NER.get_model()

            # Interactive
            text = input("insert text:")

            # Identify mentions
            samples = _annotate(ner_model, [text])

            _print_colorful_text(text, samples)

        else:
            if logger:
                logger.info("test dataset mode")

            if test_data:
                samples = test_data
            else:
                # Load test mentions
                samples = _get_test_samples(
                    args.test_mentions,
                    args.test_entities,
                    title2id,
                    wikipedia_id2local_id,
                    logger,
                )

            stopping_condition = True

        # don't look at labels
        # keep_all = (
        #     args.interactive
        #     or samples[0]["label"] == "unknown"
        #     or int(samples[0]["label_id"]) < 0
        # )
        keep_all = True

        # prepare the data for biencoder
        if logger:
            logger.info("preparing data for biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        if logger:
            logger.info("run biencoder")
        print("run biencoder")
        top_k = args.top_k
        # Checkpoints (added for resa2el): the bi-encoder top-k and each
        # crossencoder chunk are saved in ckpt_dir, so a stopped run
        # continues where it was.
        ckpt_dir = None if args.interactive else getattr(args, "ckpt_dir", None)
        bienc_path = os.path.join(ckpt_dir, "biencoder.npz") if ckpt_dir else None
        if ckpt_dir:
            _ckpt_meta(ckpt_dir, {
                "n": len(samples), "top_k": top_k,
                "chunk": getattr(args, "cross_chunk", None) or len(samples),
                "first": str(samples[0].get("mention", ""))[:50],
                "last": str(samples[-1].get("mention", ""))[:50],
            })
        if bienc_path and os.path.exists(bienc_path):
            saved = np.load(bienc_path)
            labels, nns, scores = list(saved["labels"]), list(saved["nns"]), list(saved["scores"])
            nn_data = None
            assert len(nns) == len(samples), bienc_path
            print("biencoder: loaded top-{} of {} mentions from {}".format(top_k, len(nns), bienc_path), flush=True)
        else:
            labels, nns, scores, nn_data = _run_biencoder(
                biencoder, dataloader, candidate_encoding, top_k, faiss_indexer, device, cand_pool_path=args.candidate_pool, id2title=id2title, save_topk_result=args.save_topk_result
            )
            if bienc_path:
                tmp_path = bienc_path + ".tmp.npz"
                np.savez(tmp_path, labels=np.asarray(labels), nns=np.asarray(nns), scores=np.asarray(scores))
                os.replace(tmp_path, bienc_path)

        if args.save_topk_result and nn_data is not None:
            save_data_path = os.path.join(args.output_path, "%s.t7" % args.mode)
            torch.save(nn_data, save_data_path)

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            _print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1

            if args.fast:
                # use only biencoder
                continue

        else:

            biencoder_accuracy = -1
            recall_at = -1

            if args.fast:

                predictions = []
                for entity_list in nns:
                    sample_prediction = []
                    for e_id in entity_list:
                        e_title = id2title[e_id]
                        sample_prediction.append(e_title)
                    predictions.append(sample_prediction)

                # use only biencoder
                return (
                    biencoder_accuracy,
                    recall_at,
                    -1,
                    -1,
                    len(samples),
                    predictions,
                    scores,
                )
        # prepare crossencoder data and run the crossencoder.
        # In low-memory mode this runs on chunks of mentions; the result is
        # the same because each mention is scored on its own.
        n = len(samples)
        chunk = getattr(args, "cross_chunk", None) or n
        amp = getattr(args, "amp", None)
        amp = None if amp in (None, "none") else amp
        max_seq_length = crossencoder_params["max_seq_length"]

        def _prep(c0, c1):
            context_input, candidate_input, label_input = prepare_crossencoder_data(
                crossencoder.tokenizer, samples[c0:c1], labels[c0:c1], nns[c0:c1], id2title, id2text, id2hyper, keep_all=True, args=crossencoder_params
            )
            return _modify_fast(context_input, candidate_input, max_seq_length), label_input

        def _score(prepared):
            context_input, label_input = prepared
            dataloader = _process_crossencoder_dataloader(
                context_input, label_input, crossencoder_params
            )
            # run crossencoder and get accuracy
            accuracy, part_index, part_scores = _run_crossencoder(
                crossencoder,
                dataloader,
                logger,
                context_len=biencoder_params["max_context_length"],
                amp=amp,
            )
            return part_index, part_scores

        index_array, unsorted_scores = _score_in_chunks(
            n, chunk, _prep, _score, ckpt_dir=ckpt_dir, prefetch=getattr(args, "prefetch", True),
            tag="amp={}".format(amp or "none"),
        )

        if args.interactive:

            print("\naccurate (crossencoder) predictions:")

            _print_colorful_text(text, samples)

            # print crossencoder prediction
            idx = 0
            for entity_list, index_list, sample in zip(nns, index_array, samples):
                e_id = entity_list[index_list[-1]]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1

        else:

            scores = []
            predictions = []
            for entity_list, index_list, scores_list in zip(
                nns, index_array, unsorted_scores
            ):

                index_list = index_list.tolist()

                # descending order
                index_list.reverse()

                sample_prediction = []
                sample_scores = []
                for index in index_list:
                    e_id = entity_list[index]
                    e_title = id2title[e_id]
                    sample_prediction.append(e_title)
                    sample_scores.append(scores_list[index])
                predictions.append(sample_prediction)
                scores.append(sample_scores)

            crossencoder_normalized_accuracy = -1
            overall_unormalized_accuracy = -1
            if not keep_all:
                crossencoder_normalized_accuracy = accuracy
                print(
                    "crossencoder normalized accuracy: %.4f"
                    % crossencoder_normalized_accuracy
                )

                if len(samples) > 0:
                    overall_unormalized_accuracy = (
                        crossencoder_normalized_accuracy * len(label_input) / len(samples)
                    )
                print(
                    "overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy
                )
            return (
                biencoder_accuracy,
                recall_at,
                crossencoder_normalized_accuracy,
                overall_unormalized_accuracy,
                len(samples),
                predictions,
                scores,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="Test Dataset."
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="Test Entities."
    )

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of candidates retrieved by biencoder.",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="whether to show entity url in interactive mode",
    )

    parser.add_argument(
        "--faiss_index", type=str, default=None, help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)

    models = load_models(args, logger)
    run(args, logger, *models)