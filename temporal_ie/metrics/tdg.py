import logging

import numpy as np

from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.distributed as dist

class TDGGraphMetrics:
    def __init__(self, args):
        self.args = args
        self.relation_type = args.relation_type

    def __call__(self, all_outputs, all_labels, all_label_ids):
        graphs, gold_graphs = build_graphs(all_outputs, all_labels, all_label_ids, remove_none_edges=False)

        results = {}
        results['num_articles'] = len(graphs)
        results['num_edges'] = sum(len(x) for x in graphs.values())

        results['pairwise-f1'] = pairwise(lambda x,y: f1_score(x,y,average='micro'), all_outputs, all_labels)
        results['pairwise-acc'] = pairwise(accuracy_score, all_outputs, all_labels)

        results['graph-f1-labelled'] = graph_eval(
            all_outputs, all_labels, all_label_ids,
            labelled=True, prune=False)
        results['graph-f1-unlabelled'] = graph_eval(
            all_outputs, all_labels, all_label_ids,
            labelled=False, prune=False)
        results['graph-f1-labelled-pruned'] = graph_eval(
            all_outputs, all_labels, all_label_ids,
            labelled=True, prune=self.relation_type)
        results['graph-f1-unlabelled-pruned'] = graph_eval(
            all_outputs, all_labels, all_label_ids,
            labelled=False, prune=self.relation_type)
        return results

class RankingGraphMetrics:
    def __init__(self):
        pass

    def __call__(self, all_outputs, all_labels, all_label_ids):
        results = {}

        tp, fp, fn = ranking_f1(all_outputs, all_labels, all_label_ids)
        f1_parts = torch.tensor([tp, fp, fn]).to(all_outputs.device)

        dist.all_reduce(f1_parts, op=dist.ReduceOp.SUM)
        tp, fp, fn = f1_parts
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2 * precision * recall / (precision + recall)

        results = {'joint-unlabelled-f1': f1.item()}
        results['joint-unlabelled-parts'] = f1_parts.tolist()
        results['joint-unlabelled-precision'] = precision.item()
        results['joint-unlabelled-recall'] = recall.item()

        return results

def ranking_f1(all_outputs, all_labels, all_label_ids):

    label_id_offset = 0
    pred_edges, gold_edges = [], []

    for edge_output, edge_label in zip(all_outputs, all_labels):
        gold_idx = edge_label.item()
        num_edges = sum(edge_output != -float('inf'))
        label_ids = all_label_ids[label_id_offset:label_id_offset+num_edges]

        pred_idx = torch.argmax(edge_output).item()
        pred_edges.append(tuple(label_ids[pred_idx].tolist()))
        gold_edges.append(tuple(label_ids[gold_idx].tolist()))

        label_id_offset += num_edges

        assert label_ids[gold_idx][2].item() != 0

    return set_f1_parts(pred_edges, gold_edges)

def joint_graph_eval(all_outputs, all_labels, all_label_ids):
    articles = organize_articles(all_outputs, all_labels, all_label_ids)

def organize_articles(all_outputs, all_labels, all_label_ids):
    articles = {}
    probs = torch.nn.functional.softmax(all_outputs, dim=1)

    for prob, label, id_tensor in zip(probs, all_labels, all_label_ids):
        article_id, rel_type, cs, cb, ce, ps, pb, pe = id_tensor.tolist()

        if article_id not in articles:
            articles[article_id] = {0: {}, 1:{}, 2:{}}
        article = articles[article_id]

        child, parent = (cs, cb, ce), (ps, pb, pe)
        if child not in article[rel_type]:
            article[rel_type][child] = []
        node_lst = article[rel_type][child]
        node_lst.append((parent, prob.item(), label))
    return articles

def pairwise(score_func, all_outputs, all_labels):
    _, predictions = torch.max(all_outputs.detach().cpu(), dim=1)
    all_labels = all_labels.detach().cpu()
    return score_func(all_labels, predictions)

def build_graphs(outputs, labels, label_ids, remove_none_edges=True):

    graphs = {}
    gold_graphs = {}

    probs = torch.nn.functional.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, dim=1)
    for prob, prediction, label, id_tensor in zip(probs, predictions, labels, label_ids):

        gold_label = label.item()
        article_id, s1, b1, e1, s2, b2, e2 = id_tensor.tolist()
        pred_label = prediction.item()

        edge = (article_id, (s1, b1, e1), (s2, b2, e2), pred_label, prob[pred_label].item())
        gold_edge = (article_id, (s1, b1, e1), (s2, b2, e2), gold_label)

        if article_id in graphs:
            graphs[article_id].append(edge)
            gold_graphs[article_id].append(gold_edge)
        else:
            graphs[article_id] = [edge]
            gold_graphs[article_id]= [gold_edge]

    if remove_none_edges:
        pruned_graphs = {}
        pruned_gold_graphs = {}
        for article_id, edges in graphs.items():
            edges = [x for x in edges if x[-2] != 0]
            pruned_graphs[article_id] = edges
        for article_id, edges in gold_graphs.items():
            edges = [x for x in edges if x[-1] != 0]
            pruned_gold_graphs[article_id] = edges
        return pruned_graphs, pruned_gold_graphs
    else:
        return graphs, gold_graphs

def get_edges(graphs, labelled=True):
    edges = []
    for _, graph_edges in graphs.items():
        for edge in graph_edges:
            article_id, v1, v2, label, *_ = edge
            if labelled:
                edges.append((article_id, v1, v2, label))
            else:
                edges.append((article_id, v1, v2))
    return edges

def t2t_prune_graphs(graph):
    # Assume empty edges (None label) are already pruned
    children = {}
    for edge in graph:
        _, v1, v2, label, prob = edge
        assert label != 0, 'Prune empty edges (None label) first'
        if v1 in children:
            _, _, _, _, prob2 = children[v1]
            if prob > prob2:
                children[v1] = edge
        else:
            children[v1] = edge

    edges = list(children.values())
    return edges

def e2t_prune_graphs(graph):
    children, children_dep = {}, {}
    for edge in graph:
        _, v1, v2, label, prob = edge
        assert label != 0, 'Prune empty edges (None label) first'

        if label == 1:
            c_dict = children_dep
        else:
            c_dict = children

        if v1 in c_dict:
            _, _, _, _, prob2 = c_dict[v1]
            if prob > prob2:
                c_dict[v1] = edge
        else:
            c_dict[v1] = edge

    edges = list(children.values()) + list(children_dep.values())
    return edges

def e2e_prune_graphs(graph):
    children, children_dep = {}, {}
    for edge in graph:
        _, v1, v2, label, prob = edge
        assert label != 0, 'Prune empty edges (None label) first'

        if label == 1:
            c_dict = children_dep
        else:
            c_dict = children

        if v1 in c_dict:
            _, _, _, _, prob2 = c_dict[v1]
            if prob > prob2:
                c_dict[v1] = edge
        else:
            c_dict[v1] = edge

    edges = list(children.values()) + list(children_dep.values())
    return edges

def prune_graphs(graphs, relation_type):
    prune = {
        't2t': t2t_prune_graphs,
        'e2t': e2t_prune_graphs,
        'e2e': e2e_prune_graphs,
    }[relation_type]
    result_graphs = {}
    for a_id, graph in graphs.items():
        result_graphs[a_id] = prune(graph)
    return result_graphs

def set_f1_parts(lst1, lst2):
    set1, set2 = set(lst1), set(lst2)

    tp = len(set1.intersection(set2))
    fp = len(set1.difference(set2))
    fn = len(set2.difference(set1))
    return tp, fp, fn

def set_f1(lst1, lst2):
    set1, set2 = set(lst1), set(lst2)

    tp = len(set1.intersection(set2))
    fp = len(set1.difference(set2))
    fn = len(set2.difference(set1))

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    if precision + recall == 0:
        return 0.
    return 2 * precision * recall / (precision + recall)

def graph_eval(outputs, labels, label_ids,
    labelled=False, prune=False):
    pred_graphs, gold_graphs = build_graphs(outputs, labels, label_ids, remove_none_edges=True)
    if prune:
        pred_graphs = prune_graphs(pred_graphs, prune)
    pred_edges = get_edges(pred_graphs, labelled=labelled)
    gold_edges = get_edges(gold_graphs, labelled=labelled)
    f1 = set_f1(pred_edges, gold_edges)
    return f1
