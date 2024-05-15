r""" some utils function used for retrieve """
import numpy as np
import torch
import faiss
import numpy
def retrieve_k_nearest(query, faiss_index, k):
    r"""
    use faiss to retrieve k nearest item
    """
    query_shape = list(query.size())

    # TODO: i dont know why can't use view but must use reshape here
    distances, indices = faiss_index.search(
                        query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy(), k)
    distances = torch.tensor(distances, device=query.device).view(*query_shape[:-1], k)
    indices = torch.tensor(indices, device=query.device).view(*query_shape[:-1], k)
    # print(1, distances, indices)
    return {"distances": distances, "indices": indices}

def retrieve_k_nearest_idx(query, faiss_index, k, idx, prod):
    r"""
    use faiss to retrieve k nearest item
    """
    query_shape = list(query.size())

    # TODO: i dont know why can't use view but must use reshape here


    subset = idx
    # sel = faiss.IDSelectorBatch(
    #     len(subset),
    #     faiss.swig_ptr(subset)
    # )
    # params = faiss.IVFSearchParameters()
    # params.sel = sel
    # distances, indices = faiss_index.search(
    #                     query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy(), k, params=params)
    # distances = torch.tensor(distances, device=query.device).view(*query_shape[:-1], k)
    # indices = torch.tensor(indices, device=query.device).view(*query_shape[:-1], k)
    distances, indices = faiss_index.search(
                        query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy(), k*3)
    # print(distances, indices)
    # print(indices)
    indices_ = [id for i, id in enumerate(indices[0]) if subset[id] in prod and i < k]
    distances_ = [dis for i, dis in enumerate(distances[0]) if subset[indices[0][i]] in prod and i < k]
    # for i, id in enumerate(indices[0]):
    #     if subset[id] not in prod:
    #         print(prod)
    #         print(subset[id])
    while len(indices_) < k:
        indices_.append(-1)
        distances_.append(1e20)
    # print(distances_, indices_)
    distances = torch.tensor(distances_, device=query.device).view(*query_shape[:-1], k)
    indices = torch.tensor(indices_, device=query.device).view(*query_shape[:-1], k)
    return {"distances": distances, "indices": indices}