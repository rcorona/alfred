import torch
import subprocess

def delete_keys_from_dict(dict_del, lst_keys):
    """
    Delete the keys present in lst_keys from the dictionary.
    Loops recursively over nested dictionaries.
    """
    dict_foo = dict_del.copy()  #Used as iterator to avoid the 'DictionaryHasChanged' error
    for field in dict_foo.keys():
        if field in lst_keys:
            del dict_del[field]
        if type(dict_foo[field]) == dict:
            delete_keys_from_dict(dict_del[field], lst_keys)
    return dict_del


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_partial_model(pretrained_dict, model):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def safe_zip(*seqs):
    for x in seqs[1:]:
        assert len(x) == len(seqs[0])
    return zip(*seqs)

def print_git_info():
    # print the git revision and any changes, to make self-replication easier!
    subprocess.call("git rev-parse HEAD", shell=True)
    subprocess.call("git --no-pager diff", shell=True)

def dropout_mask_like(tensor, dropout_p):
    return torch.distributions.Bernoulli(
        torch.full_like(tensor, 1.0 - dropout_p)
    ).sample().float() / (1.0 - dropout_p)

def dropout_mask(size, dropout_p):
    return torch.distributions.Bernoulli(
        torch.full(size, 1.0 - dropout_p)
    ).sample().float() / (1.0 - dropout_p)
