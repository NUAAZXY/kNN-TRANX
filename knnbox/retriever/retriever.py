import torch
from knnbox.retriever.utils import retrieve_k_nearest, retrieve_k_nearest_idx
import numpy as np
class Retriever:
    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None


    def retrieve(self, query, return_list = ["vals", "distances"], k = None, prod=None):
        r""" 
        retrieve the datastore, save and return results 
        if parameter k is provided, it will suppress self.k
        """

        k = k if k is not None else self.k
        # load the faiss index if haven't loaded
        if not hasattr(self.datastore, "faiss_index") or \
                    self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys", move_to_gpu=True)
        # self.datastore['indices']
        query = query.detach()
        if prod is not None:
            idx = np.array(self.datastore['vals'].data)
            # candidates = []
            # for p in prod:
            #     id = np.where(idx == p)[0]
            #     candidates.extend(id)
            # print(prod)
            # print(candidates)
            # print(candidates)
            # print(len(candidates))
            # candidates = np.array(candidates)
            faiss_results = retrieve_k_nearest_idx(query, self.datastore.faiss_index["keys"], k, idx, prod)
        else:
            faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], k)

        ret = {}
        if "distances" in return_list:
            ret["distances"] = faiss_results["distances"]
        if "indices" in return_list:
            ret["indices"] = faiss_results["indices"]
        if "k" in return_list:
            ret["k"] = k
        if "query" in return_list:
            ret["query"] = query

        # other information get from self.datastores.datas using indices, for example `keys` and `vals`
        indices = faiss_results["indices"].cpu().numpy()
        for data_name in return_list:
            if data_name not in ["distances", "indices", "k", "query"]:
                assert data_name in self.datastore.datas, \
                                    "You must load the {} of datastore first".format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        self.results = ret # save the retrieved results
        return ret
    
        