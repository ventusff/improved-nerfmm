import os
import urllib
import torch
from torch.utils import model_zoo
# torch.autograd.set_detect_anomaly(True)

class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''

    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        '''Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        '''
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename)

    def load_file(self, filepath, no_reload=False, ignore_keys=[], only_use_keys=None):
        '''Loads a module dictionary from file.

        Args:
            filepath (str): file path of saved module dictionary
        '''

        assert not ((len(ignore_keys) > 0) and only_use_keys is not None), \
            'please specify at most one in [ckpt_ignore_keys, ckpt_only_use_keys]'

        if filepath is not None and filepath != "None":
            ckpts = [filepath]
        else:
            ckpts = sorted_ckpts(self.checkpoint_dir)

        print("=> Found ckpts:",
              "{}".format(ckpts) if len(ckpts) < 5 else "...,{}".format(ckpts[-5:]))

        if len(ckpts) > 0 and not no_reload:
            ckpt_file = ckpts[-1]
            print('=> Loading checkpoint from local file', ckpt_file)
            state_dict = torch.load(ckpt_file)

            if len(ignore_keys) > 0:
                to_load_state_dict = {}
                for k in state_dict.keys():
                    if k in ignore_keys:
                        print("=> [ckpt] Ignoring keys: {}".format(k))
                    else:
                        to_load_state_dict[k] = state_dict[k]
            elif only_use_keys is not None:
                if not isinstance(only_use_keys, list):
                    only_use_keys = [only_use_keys]
                to_load_state_dict = {}
                for k in only_use_keys:
                    print("=> [ckpt] Only use keys: {}".format(k))
                    to_load_state_dict[k] = state_dict[k]
            else:
                to_load_state_dict = state_dict

            scalars = self.parse_state_dict(to_load_state_dict, ignore_keys)
            return scalars
        else:
            return {}

    def load_url(self, url):
        '''Load a module dictionary from url.

        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict, ignore_keys):
        '''Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
    '''

        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                if k not in ignore_keys:
                    print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {k: v for k, v in state_dict.items()
                   if k not in self.module_dict}
        return scalars


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')


def sorted_ckpts(checkpoint_dir):
    ckpts = []
    if os.path.exists(checkpoint_dir):
        latest = None
        final = None
        ckpts = []
        for fname in sorted(os.listdir(checkpoint_dir)):
            fpath = os.path.join(checkpoint_dir, fname)
            if ".pt" in fname:
                ckpts.append(fpath)
                if 'latest' in fname:
                    latest = fpath
                elif 'final' in fname:
                    final = fpath
        if latest is not None:
            ckpts.remove(latest)
            ckpts.append(latest)
        if final is not None:
            ckpts.remove(final)
            ckpts.append(final)
    return ckpts