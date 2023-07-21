from .tdg import TDGGraphMetrics

metrics = {
    'tdg': TDGGraphMetrics,
}

def setup_metrics(args):
    return metrics[args.dataset_type](args)