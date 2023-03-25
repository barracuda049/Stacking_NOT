import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.data import TensorDataset
from .distributions import LoaderSampler

def transform_data(sampler, save_path, model): # device='cuda'
    """
    This function is used to save mapped distribution

    Args:
        sampler (LoaderSampler): input distribution that we want to map. Example: X_sampler
        save_path (str): path where the mapped distribution will be saved
        model (UNet): trained model 
    """

    loader = sampler.loader

    preds = []
    with torch.no_grad():
        for batch, _ in loader:
            # batch = batch.to(device)
            pred = model(batch)
            preds.append(pred)

    new_sampler = torch.cat(preds, dim=0)

    torch.save(new_sampler, save_path)
    print('Done!')

def new_sample_the_same(path_to_new_data, batch_size=64, device='cuda', shuffle=False):
    """
    This function is used to load the mapped distrinution and return sampler

    Args:
        path_to_new_data (str): expected the path to saved sampler. Example: "X_sampler_new.pt"
        batch_size (int): the size of batch. Defaults to 64.
        device (str): set device. Defaults to 'cuda'.
        shuffle (bool): Set True if you want to shuffle the data. Defaults to False.

    Returns:
        LoaderSampler: return new sampler
    """
    
    loaded_new_data = torch.load(path_to_new_data)
    
    dataset = TensorDataset(loaded_new_data, torch.zeros(len(loaded_new_data)).view(-1, 1))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=shuffle, num_workers=8, batch_size=batch_size), device=device)

    return sampler


# Not use
# def new_samplers(path_to_new_data, batch_size = 64, device = 'cuda', test_ratio = 0.1):
#     loaded_new_data = []
#     for i in path_to_new_data:
#         loaded_new_data.append(torch.load(i))

#     loaded_new_data = torch.cat(loaded_new_data, dim = 0)


#     dataset = TensorDataset(loaded_new_data, torch.zeros(len(loaded_new_data)).view(-1, 1))
#     idx = list(range(len(dataset)))

#     test_size = int(len(idx) * test_ratio)
#     train_idx, test_idx = idx[:-test_size], idx[-test_size:]
#     train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

#     train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device = device)
#     test_sampler = LoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device = device)

#     return train_sampler, test_sampler