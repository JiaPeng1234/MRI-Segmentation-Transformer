import types
import yaml
import logging
import os
import tempfile
import shutil
import datetime

"""Parameter file"""


def fetch_options(idx_dataset=0):
    """Define default values for parameters used within tf"""
    op = types.SimpleNamespace()

    # setup
    op.device = '0'  # standard device, can be bypassed if device is provided during call

    # i/o
    op.path_parser_cfg = '/home/io_patterns_nako.yaml'

    if idx_dataset == 0:
        op.dir_data = '/home/NAKO/'
    else:
        raise ValueError('Chosen dataset idx is not available')

    op.set_split = 70

    # pipeline & augmentation
    op.shape_image = [320, 260, 316]
    op.shape_image_eval = [160, 160, 160]  # smaller image results in faster eval time (less patches) old:[240, 240, 240], i have changed it to [160, 160, 160]
    op.shape_input = [64, 64, 64]  # has to correspond to your designed model
    op.shape_output = [16, 16, 16]  # has to correspond to your designed model
    op.size_batch = 24
    op.size_batch_eval = 50
    op.size_buffer = 1
    op.num_parallel_calls = 4
    op.repeat = 5
    op.b_shuffle = True
    op.b_eval_labels_patch = False
    op.b_eval_labels_image = True

    op.patches_per_class = [2, 1, 1, 1, 1, 1]  # Note: atm has to be reflected in your input pipeline #2

    # patch augmentation
    op.sigma_offset = 0.1
    op.sigma_noise = 0.05
    op.sigma_pos = 0.08

    # image augmentation
    op.b_mirror = False
    op.b_rotate = True
    op.b_scale = True
    op.b_warp = False
    op.b_permute_labels = False
    op.angle_max = 7
    op.scale_factor = 0.08
    op.delta_max = 0

    # model
    op.channels = 2
    op.channels_out = 6
    op.b_dynamic_pos_mid = True
    op.b_dynamic_pos_end = False
    op.filters = 32
    op.dense_layers = 2
    op.alpha = 0.2
    op.rate_dropout = 0.0

    # optimizer
    op.rate_learning = 0.00001  # std: 1e-2 - 1e-6, too low: slow learning
    op.beta1 = 0.9  # std: 0.9
    op.beta2 = 0.999  # std: 0.999
    op.epsilon = 0.00000001  # std: 1e-8, too high: slow learning

    # session
    op.num_epochs = 701
    op.b_continuous_eval = True
    op.b_restore = False
    op.save_summary_steps = 1
    op.b_viewer_train = False
    op.b_viewer_eval = False
    op.b_save_pred = False

    # logging
    op.log_level = logging.INFO

    # seed
    op.b_use_seed = False
    op.random_seed = 100

    return op


class Params(object):

    def __init__(self, path_yaml='', model_dir='', idx_dataset=-1, b_recreate=False):

        # set yaml path:
        self.path_yaml = path_yaml

        # fetch dataset idx to choose hardcoded split
        self.idx_dataset = idx_dataset

        # add passed/generated params file
        self.update(b_recreate)

        # add default values
        if model_dir:
            self.set_model_dir(model_dir)

    def create(self):
        # Create new default params
        logging.info('Creating a new set of parameters in %s', self.path_yaml)
        self.__dict__.update(fetch_options(idx_dataset=self.idx_dataset).__dict__)
        self.save()

    def save(self):
        if not self.path_yaml:
            _, self.path_yaml = tempfile.mkstemp()
        try:
            with open(self.path_yaml, 'w') as file:
                yaml.dump(self.__dict__, file, indent=4)
        except Exception:
            os.remove(self.path_yaml)
            raise Exception()

    def update(self, b_recreate):
        if b_recreate or not os.path.isfile(self.path_yaml):
            self.create()

        with open(self.path_yaml, 'r') as file:
            params = yaml.load(file)
            self.__dict__.update(params)

    def set_path(self, path_yaml):
        self.path_yaml = path_yaml

    def set_model_dir(self, model_dir):
        self.__dict__['date'] = datetime.datetime.now().strftime('_%Y-%m-%dT%H-%M-%S')
        self.__dict__['dir_model'] = os.path.join(model_dir, 'run' + self.__dict__['date'])
        self.__dict__['dir_logs_train'] = os.path.join(self.__dict__['dir_model'], 'logs', 'train')
        self.__dict__['dir_logs_eval'] = os.path.join(self.__dict__['dir_model'], 'logs', 'eval')
        self.__dict__['dir_graphs_train'] = os.path.join(self.__dict__['dir_model'], 'graphs', 'train')
        self.__dict__['dir_graphs_eval'] = os.path.join(self.__dict__['dir_model'], 'graphs', 'eval')
        self.__dict__['dir_ckpts'] = os.path.join(self.__dict__['dir_model'], 'ckpts')
        self.__dict__['dir_ckpts_best'] = os.path.join(self.__dict__['dir_model'], 'ckpts_best')

        # Create environment
        for k, v in self.__dict__.items():
            if 'dir' in k:
                if not os.path.exists(v):
                    os.makedirs(v, exist_ok=True)

        # Move params to corresponding model folder
        self.move(os.path.join(self.__dict__['dir_model'], 'params.yaml'))

        # Save with dirs
        self.save()

    def move(self, path_yaml_new):
        shutil.move(self.path_yaml, path_yaml_new)
        self.path_yaml = path_yaml_new

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
