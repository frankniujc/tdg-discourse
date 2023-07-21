from .tdg import tdg_baseline_trainloop, tdg_spe_trainloop

models = {
    'baseline': tdg_baseline_trainloop,
    'spe': tdg_spe_trainloop,
}

def setup_model(args, output_size):
    model_cls, training_loop, setup_training = models[args.model_arch]
    return model_cls.from_pretrained(args.model_name_or_path, args, output_size), training_loop, setup_training
