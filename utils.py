import matplotlib.pyplot as plt
import yaml


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]

def save_yaml_train(args, yaml_name):
    para = {'epochs': 0,
    'GPU': 0,
    'batch_size': 0,
    'datasets_folder': 0,
    'datasets_path': 0,
    'output_path': 0,
    'pth_path': 0,
    'patch_x': 0,
    'patch_y': 0,
    'gap_y': 0,
    'gap_x': 0,
    'burst': 0,
    'max_lr': 0,
    'min_lr': 0,
    'b1': 0,
    'b2': 0}
    para["epochs"] = args.epochs
    para["GPU"] = args.GPU
    para["batch_size"] = args.batch_size
    para["datasets_folder"] = args.datasets_folder
    para["datasets_path"] = args.datasets_path
    para["output_path"] = args.output_path
    para["pth_path"] = args.pth_path
    para["patch_x"] = args.patch_x
    para["patch_y"] = args.patch_y
    para["burst"] = args.burst
    para["gap_x"] = args.gap_x
    para["gap_y"] = args.gap_y
    para["max_lr"] = args.max_lr
    para["min_lr"] = args.min_lr
    para["b1"] = args.b1
    para["b2"] = args.b2
    with open(yaml_name, 'w') as f:
        data = yaml.dump(para, f)
        
def save_yaml_test(args, yaml_name):
    para = {
        'datasets_path':0,
        'datasets_folder':0,
        'denoise_model':0,
        'plan_path':0,
        'output_path':0,
        'GPU':0,
        'batch_size':0,
        'patch_x':0,
        'patch_y':0,
        'patch_t':0,
        'gap_x':0,
        'gap_y':0,
        'gap_t':0,
        'test_datasize':0,
        'scale_factor':0
    }
    para["datasets_path"] = args.datasets_path
    para["datasets_folder"] = args.datasets_folder
    para["denoise_model"] = args.denoise_model
    # para["plan_path"] = args.plan_path
    para["output_path"] = args.output_path
    para["GPU"] = args.GPU
    para["batch_size"] = args.batch_size
    para["patch_x"] = args.patch_x
    para["patch_y"] = args.patch_y
    para["burst"] = args.burst
    para["gap_x"] = args.gap_x
    para["gap_y"] = args.gap_y
    para["scale_factor"] = args.scale_factor
    with open(yaml_name, 'w') as f:
        data = yaml.dump(para, f)