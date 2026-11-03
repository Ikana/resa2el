import json
import pickle
import blink.main_dense as main_dense
import argparse
import pickle
import sys, io, os
import requests
import numpy as np
import argparse


titles = {}
hyperlinks2t = {}
title_text = {}

clusters = {}
spacy_pages = {}


def GetWikiID(id2title, title):
	title = title.replace(" ", "_")
	if title in id2title[1]:
		return id2title[1][title]
	else:
		if title[0] == "'" and title[-1] == "'":
			title_form = title[1:-1]
			return id2title[1][title_form]
	return ''


def BuildLowMem(in_args):
	# Build the low-memory store once (needs about 70 GB of RAM).
	from blink import lowmem
	id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
	id2title = pickle.load(id2title_file)
	print('ID to title loaded.')
	text_file = open('data/wikipedia/wiki/title_text.json', 'r')
	title_text = json.load(text_file)
	print("Text loaded.")
	hyper_file = open('data/wikipedia/wiki/t2hyperlinks.json', 'r')
	t2h = json.load(hyper_file)
	print("Hyperlinks loaded.")
	n = lowmem.build_store(id2title, title_text, t2h, in_args.lowmem)
	print("Store written:", n, "candidates")
	del id2title, title_text, t2h
	shape, dtype = lowmem.convert_encoding("models/all_entities.encoding", in_args.lowmem)
	print("Encoding written:", shape, dtype)


def EvaluateBLINK(in_args):
	if in_args.lowmem:
		LowMemEvaluate(in_args)
		return

	id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
	id2title = pickle.load(id2title_file)
	print('ID to title loaded.')

	text_file = open('data/wikipedia/wiki/title_text.json', 'r')
	title_text = json.load(text_file)
	print("Text loaded.")

	hyper_file = open('data/wikipedia/wiki/t2hyperlinks.json', 'r')
	t2h = json.load(hyper_file)
	print("Hyperlinks loaded.")

	localid2title = {}
	localid2text = {}
	localid2hyperlinks = {}
	idx = 0
	for title in title_text:
		localid2title[idx] = title
		localid2text[idx] = title_text[title][0:2000]
		localid2hyperlinks[idx] = []

		title_form = title
		if title[0] == "'" and title[-1] == "'":
			title_form = title[1:-1]

		if title_form.replace(" ", '_') in id2title[1] and id2title[1][title_form.replace(" ", '_')] in t2h:
			for t_idx, linked_title in enumerate(t2h[id2title[1][title_form.replace(" ", '_')]]):
				if t_idx > 10:
					break
				link = t2h[id2title[1][title_form.replace(" ", '_')]][linked_title]
				if link['end'] > 2000:
					continue
				localid2hyperlinks[idx].append((linked_title, title_text[title][link['start']: link['end']]))
		idx += 1

	config = MakeConfig(in_args)

	RunBLINK(in_args, config, localid2title, localid2text, localid2hyperlinks)


def LowMemEvaluate(in_args):
	# Same as EvaluateBLINK, but the candidate maps come from the store
	# made by BuildLowMem, and are read one candidate at a time.
	from blink import lowmem
	store = lowmem.LowMemStore(in_args.lowmem)
	print("Low-memory store opened:", len(store), "candidates")
	config = MakeConfig(in_args)
	config["lowmem_dir"] = in_args.lowmem
	RunBLINK(in_args, config, store.title, store.text, store.hyper)


def MakeConfig(in_args):
	return {
		"test_entities": None,
		"test_mentions": None,
		"interactive": False,
		"top_k": in_args.topk,
		"biencoder_model": "models/biencoder/pytorch_model.bin",
		"biencoder_config": "models/biencoder/params.json",
		"entity_catalogue": "models/entity.jsonl",
		"entity_encoding": "models/all_entities.encoding",
		"candidate_pool": "models/cand_pool.t7",
		"crossencoder_model": "models/crossencoder/pytorch_model.bin",
		"crossencoder_config": "models/crossencoder/params.json",
		"fast": in_args.fast,  # set this to be true if speed is a concern
		"output_path": "out/",  # logging directory
		"mode": in_args.mode,
		"save_topk_result": in_args.save_topk_result,
		"amp": in_args.amp,
		"ckpt_dir": in_args.ckpt,
		"prefetch": not in_args.no_prefetch,
		"cross_chunk": in_args.cross_chunk,
		"cross_batch": in_args.cross_batch,
	}


def OutSuffix(in_args):
	# --limit N runs the first N mentions only (timing tests). Its output
	# files get a suffix, so they are never mixed up with a full run.
	return "_limit{}".format(in_args.limit) if getattr(in_args, "limit", None) else ""


def RunBLINK(in_args, config, localid2title, localid2text, localid2hyperlinks):
	args = argparse.Namespace(**config)
	models = main_dense.load_models(args, logger=None)
	print("Model loaded.")

	data_to_link = []

	if in_args.nyt:
		with io.open('data/nyt/nyt_test.jsonl', mode="r", encoding="utf-8") as file:
			for line in file:
				data_to_link.append(json.loads(line.strip()))
		if in_args.limit:
			data_to_link = data_to_link[:in_args.limit]
		_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link, device='cuda',
		                                                     alternate_mapping=(
		                                                     localid2title, localid2text, localid2hyperlinks))
		for idx, data in enumerate(data_to_link):
			data['edl'] = predictions[idx]
			data['scores'] = np.array(scores[idx]).tolist()
		with open(args.output_path + 'nyt_test{}.jsonl'.format(OutSuffix(in_args)), 'w') as outfile:
			for entry in data_to_link:
				json.dump(entry, outfile)
				outfile.write('\n')
	else:
		infile = open("data/wikipedia/preprocessed/wiki_{}.json".format(args.mode), 'r')
		clusters = json.load(infile)

		with io.open('out/{}.jsonl'.format(args.mode), mode="r", encoding="utf-8") as file:
			for line in file:
				data_to_link.append(json.loads(line.strip()))
		if in_args.limit:
			data_to_link = data_to_link[:in_args.limit]

		_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link, device='cuda',
		                                                     alternate_mapping=(localid2title, localid2text, localid2hyperlinks))
		idx = 0
		out_clusters = {}  # with --limit: only the mentions that have a prediction
		for t_idx, title in enumerate(clusters):
			mentions = []
			for mention in clusters[title]:
				if in_args.limit and idx >= len(predictions):
					break
				mention['edl'] = predictions[idx]
				mention['scores'] = np.array(scores[idx]).tolist()
				mentions.append(mention)
				idx += 1
			if mentions:
				out_clusters[title] = mentions
		assert idx == len(predictions), (idx, len(predictions))

		outfile = open(args.output_path + '{}_result_top{}{}.json'.format(args.mode, args.top_k, OutSuffix(in_args)), 'w')
		json.dump(out_clusters if in_args.limit else clusters, outfile)


def Ratio(correct, total):
	# With --limit a class can be empty. Then its accuracy is not defined.
	return float(correct) / float(total) if total else float("nan")


def WikiAccuracy(args, is_recall=False):
	infile = open('out/{}_result_top{}{}.json'.format(args.mode, args.topk, OutSuffix(args)), 'r')
	clusters = json.load(infile)

	id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
	id2title = pickle.load(id2title_file)

	if not args.lowmem:  # title_text is not used below
		text_file = open('data/wikipedia/wiki/title_text.json', 'r')
		title_text = json.load(text_file)

	correct = 0
	correct_v = 0
	correct_v_seen = 0
	correct_v_unseen_form = 0
	correct_v_unseen = 0
	correct_nom = 0
	correct_nom_hard = 0
	correct_nom_easy = 0

	total = 0
	total_v = 0
	total_v_seen = 0
	total_v_unseen_form = 0
	total_v_unseen = 0
	total_nom = 0
	total_nom_hard = 0
	total_nom_easy = 0

	for event in clusters:
		for m_idx, mention in enumerate(clusters[event]):
			flag = False
			for idx, title in enumerate(mention['edl']):
				id = GetWikiID(id2title, title)
				gold_id = GetWikiID(id2title, event)
				if str(id) == str(gold_id):
					correct += 1
					if mention['type'] == 'verb':
						correct_v += 1
						if mention['status'] == 'unseen_event':
							correct_v_unseen += 1
						if mention['status'] == 'seen_event_seen_form':
							correct_v_seen += 1
						if mention['status'] == 'seen_event_unseen_form':
							correct_v_unseen_form += 1
					else:
						correct_nom += 1
						if mention['status'] == 'hard':
							correct_nom_hard += 1
						if mention['status'] == 'easy':
							correct_nom_easy += 1
					flag = True
					break
				if is_recall:
					continue
				else:
					break

			total += 1
			if mention['type'] == 'verb':
				total_v += 1
				if mention['status'] == 'unseen_event':
					total_v_unseen += 1
				if mention['status'] == 'seen_event_seen_form':
					total_v_seen += 1
				if mention['status'] == 'seen_event_unseen_form':
					total_v_unseen_form += 1
			else:
				total_nom += 1
				if mention['status'] == 'hard':
					total_nom_hard += 1
				if mention['status'] == 'easy':
					total_nom_easy += 1

	print(total, total_v, total_nom)

	print("Total Mentions: ", total)
	print("Total Verb: ", total_v)
	print("Total Mom: ", total_nom)

	print("Total Verb Seen: ", total_v_seen)
	print("Total Verb Unseen Form: ", total_v_unseen_form)
	print("Total Verb Unseen", total_v_unseen)

	print("Total Nom Hard: ", total_nom_hard)
	print("Total Nom Easy: ", total_nom_easy)

	accuracy = Ratio(correct, total)
	accuracy_v = Ratio(correct_v, total_v)
	accuracy_nom = Ratio(correct_nom, total_nom)

	accuracy_v_seen = Ratio(correct_v_seen, total_v_seen)
	accuracy_v_unseen_form = Ratio(correct_v_unseen_form, total_v_unseen_form)
	accuracy_v_unseen = Ratio(correct_v_unseen, total_v_unseen)
	accuracy_nom_hard = Ratio(correct_nom_hard, total_nom_hard)
	accuracy_nom_easy = Ratio(correct_nom_easy, total_nom_easy)

	print(correct, total, accuracy, accuracy_v, accuracy_nom)
	print("Verb Seen: ", accuracy_v_seen, correct_v_seen, total_v_seen)
	print("Verb Unseen Form: ", accuracy_v_unseen_form, correct_v_unseen_form, total_v_unseen_form)
	print("Verb Unseen", accuracy_v_unseen, correct_v_unseen, total_v_unseen)

	print("Nom Hard: ", accuracy_nom_hard, correct_nom_hard, total_nom_hard)
	print("Nom Easy: ", accuracy_nom_easy, correct_nom_easy, total_nom_easy)


def callNER(text):
	NER_HTTP = "http://dickens.seas.upenn.edu:4033/ner/"

	input = {"lang": "eng", "model": "onto_ner",
	         "text": text.replace(' [ ', ' ').replace(' ] ', ' ')}
	res_out = requests.post(NER_HTTP, json=input)

	try:
		res_json = res_out.json()
		entities = []
		for i in range(len(res_json['views'])):
			if res_json['views'][i]['viewName'] == 'NER_CONLL':
				for element in res_json['views'][i]['viewData'][0]['constituents']:
					ner_type = element['label']
					surface_form = ' '.join(res_json['tokens'][element['start']: element['end']])
					entities.append({"type": ner_type, "form": surface_form})
		return entities
	except:
		return []


def NYTAccuracy(is_recall=False, suffix=""):

	test_data = []
	with io.open('out/nyt_test{}.jsonl'.format(suffix), mode="r", encoding="utf-8") as file:
		for line in file:
			test_data.append(json.loads(line.strip()))

	id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
	id2title = pickle.load(id2title_file)
	print('ID to title loaded.')

	correct = 0
	correct_v = 0
	correct_v_seen = 0
	correct_v_unseen_form = 0
	correct_v_unseen = 0
	correct_v_nil = 0
	correct_nom = 0
	correct_nom_hard = 0
	correct_nom_easy = 0
	correct_nom_nil = 0

	total = 0
	total_v = 0
	total_v_seen = 0
	total_v_unseen_form = 0
	total_v_unseen = 0
	total_v_nil = 0
	total_nom = 0
	total_nom_hard = 0
	total_nom_easy = 0
	total_nom_nil = 0

	threshold = -0.1
	print("Nil threshold: ", threshold)

	for data in test_data:
		# Skip all the Nil mentions.
		# if data['label_id'] == -1:
		# 	continue

		flag = False
		for idx, title in enumerate(data['edl'][0:5]):
			if data['scores'][idx] < threshold:
				id = -1
			else:
				id = GetWikiID(id2title, title)
			gold_id = data['label_id']
			if str(id) == str(gold_id):
				correct += 1

				if data['type'] == 'verb':
					correct_v += 1
					if id == -1:
						correct_v_nil += 1
					else:
						if data['status'] == 'unseen_event':
							correct_v_unseen += 1
						if data['status'] == 'seen_event_seen_form':
							correct_v_seen += 1
						if data['status'] == 'seen_event_unseen_form':
							correct_v_unseen_form += 1
				else:
					correct_nom += 1
					if id == -1:
						correct_nom_nil += 1
					else:
						if data['status'] == 'hard':
							correct_nom_hard += 1
						if data['status'] == 'easy':
							correct_nom_easy += 1
				flag = True
				break
			if is_recall:
				continue
			else:
				break
		total += 1
		if data['type'] == 'verb':
			total_v += 1
			if data['label_id'] == -1:
				total_v_nil += 1
			else:
				if data['status'] == 'unseen_event':
					total_v_unseen += 1
				if data['status'] == 'seen_event_seen_form':
					total_v_seen += 1
				if data['status'] == 'seen_event_unseen_form':
					total_v_unseen_form += 1
		else:
			total_nom += 1
			if data['label_id'] == -1:
				total_nom_nil += 1
			else:
				if data['status'] == 'hard':
					total_nom_hard += 1
				if data['status'] == 'easy':
					total_nom_easy += 1

	print(total, total_v, total_nom)

	print("Total Verb Seen: ", total_v_seen)
	print("Total Verb Unseen Form: ", total_v_unseen_form)
	print("Total Verb Unseen: ", total_v_unseen)
	print("Total Verb Nil: ", total_v_nil)

	print("Total Nom Hard: ", total_nom_hard)
	print("Total Nom Easy: ", total_nom_easy)
	print("Total Nom Nil: ", total_nom_nil)


	accuracy = Ratio(correct, total)
	accuracy_v = Ratio(correct_v, total_v)
	accuracy_nom = Ratio(correct_nom, total_nom)

	if total_v_seen != 0:
		accuracy_v_seen = Ratio(correct_v_seen, total_v_seen)
	else:
		accuracy_v_seen = 0
	accuracy_v_unseen_form = Ratio(correct_v_unseen_form, total_v_unseen_form)
	accuracy_v_unseen = Ratio(correct_v_unseen, total_v_unseen)
	accuracy_nom_hard = Ratio(correct_nom_hard, total_nom_hard)
	accuracy_nom_easy = Ratio(correct_nom_easy, total_nom_easy)

	accuracy_v_nil = Ratio(correct_v_nil, total_v_nil)
	accuracy_nom_nil = Ratio(correct_nom_nil, total_nom_nil)

	print(correct, total, accuracy, accuracy_v, accuracy_nom)
	print("Verb Seen: ", accuracy_v_seen, correct_v_seen, total_v_seen)
	print("Verb Unseen Form: ", accuracy_v_unseen_form, correct_v_unseen_form, total_v_unseen_form)
	print("Verb Unseen", accuracy_v_unseen, correct_v_unseen, total_v_unseen)
	print("Verb Nil", accuracy_v_nil, correct_v_nil, total_v_nil)

	print("Nom Hard: ", accuracy_nom_hard, correct_nom_hard, total_nom_hard)
	print("Nom Easy: ", accuracy_nom_easy, correct_nom_easy, total_nom_easy)
	print("Nom Nil", accuracy_nom_nil, correct_nom_nil, total_nom_nil)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--predict", action="store_true", help="Link event mentions to Wikipedia."
	)
	parser.add_argument(
		"--topk",
		dest="topk",
		type=int,
		default=10,
		help="Number of candidates retrieved by biencoder.",
	)
	parser.add_argument(
		"--fast", dest="fast", action="store_true", help="only biencoder mode"
	)
	parser.add_argument(
		"--save_topk_result",
		action="store_true",
		help="Whether to save prediction results.",
	)
	parser.add_argument(
		"--mode",
		dest="mode",
		type=str,
		default="test",
		help="data split",
	)
	parser.add_argument(
		"--evaluate", action="store_true", help="Evaluate performance"
	)
	parser.add_argument(
		"--nyt", action="store_true", help="Whether use NYT data"
	)

	parser.add_argument(
		"--lowmem", type=str, default=None,
		help="Low-memory mode: directory of the store made with --build_lowmem."
	)
	parser.add_argument(
		"--build_lowmem", action="store_true", help="Build the low-memory store in --lowmem."
	)
	parser.add_argument(
		"--cross_chunk", type=int, default=500,
		help="Low-memory mode: number of mentions per crossencoder chunk."
	)

	parser.add_argument(
		"--cross_batch", type=int, default=None,
		help="Low-memory mode: crossencoder eval batch size (default: from params.json)."
	)

	parser.add_argument(
		"--amp", choices=["none", "tf32", "fp16", "bf16"], default="none",
		help="Crossencoder precision on the GPU: none (float32), tf32, or fp16/bf16 autocast."
	)
	parser.add_argument(
		"--ckpt", type=str, default=None,
		help="Directory for bi-encoder and crossencoder checkpoints; a stopped run continues from them."
	)
	parser.add_argument(
		"--limit", type=int, default=None,
		help="Run the first N mentions only (timing tests). Output files get the suffix _limitN."
	)
	parser.add_argument(
		"--no_prefetch", action="store_true",
		help="Do not tokenize the next crossencoder chunk while the GPU scores the current one."
	)

	args = parser.parse_args()

	if args.build_lowmem:
		BuildLowMem(args)
	if args.predict:
		EvaluateBLINK(args)
	if args.evaluate:
		if args.nyt:
			NYTAccuracy(is_recall=False, suffix=OutSuffix(args))
		else:
			WikiAccuracy(args, is_recall=False)


