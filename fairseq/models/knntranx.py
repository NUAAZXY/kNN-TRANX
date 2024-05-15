from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
import torch.nn.functional as F
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask,
    archs,
    disable_model_grad,
    enable_module_grad,
)
from knnbox.datastore import Datastore, PckDatastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner, AdaptiveCombiner
# from .adaptive_knn_mt import AdaptiveKNNMT
from knnbox.models.BertranX.model.nl2code import nl2code

# @register_model("pck_knn_mt")
# class PckKNNMT(AdaptiveKNNMT):
#     r"""
#     The  pck knn-mt model.
#     """
#     @staticmethod
#     def add_args_knn(parser):
#         r"""
#         add pck knn-mt related args_knn here
#         """
#         AdaptiveKNNMT.add_args_knn(parser)
#         parser.add_argument("--knn-reduct-dim", type=int, metavar="N", default=64,
#                             help="reducted dimension of datastore")
#
#     @classmethod
#     def build_decoder(cls, args_knn, tgt_dict, embed_tokens):
#         r"""
#         we override this function, replace the TransformerDecoder with PckKNNMTDecoder
#         """
#         return KNNTRANX(
#             args_knn,
#             tgt_dict,
#             embed_tokens,
#             no_encoder_attn=getattr(args_knn, "no_cross_attention", False),
#         )

# @register_model("knntranx")
class KNNTRANX(nl2code):
    r"""
    The pck knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args_knn, parameters, act_dict, vocab, grammar, primitives_type, device, path_folder_config):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(parameters, act_dict, vocab, grammar, primitives_type, device, path_folder_config)
        self.args_knn = args_knn
        if args_knn.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = PckDatastore(
                        path=args_knn.knn_datastore_path,
                        dictionary_len=len(vocab.primitive)+len(vocab.code),
                    )  
            self.datastore = global_vars()["datastore"]
        
        else:
            self.datastore = PckDatastore.load(args_knn.knn_datastore_path, load_list=["vals"], load_network=True)
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args_knn.knn_max_k)
            if args_knn.knn_mode == "train_metak":
                # fixme:probability_dim is vague
                self.combiner = AdaptiveCombiner(max_k=args_knn.knn_max_k, probability_dim=len(vocab),
                            k_trainable=(args_knn.knn_k_type=="trainable"),
                            lambda_trainable=(args_knn.knn_lambda_type=="trainable"), lamda_=args_knn.knn_lambda,
                            temperature_trainable=(args_knn.knn_temperature_type=="trainable"), temperature=args_knn.knn_temperature
            )
            elif args_knn.knn_mode == "inference":
                self.combiner = AdaptiveCombiner.load(args_knn.knn_combiner_path)

    def forward(
        self,
        examples
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """

        if self.args_knn.knn_mode == "build_datastore":
            def get_4_gram(target):
                r"""
                args_knn:
                    target: [B, T]
                Return: [B, T, 4]
                """
                batch_size = target.size(0)
                target = target[:, :, None]
                target_pad_1 = torch.cat((torch.zeros((batch_size, 1, 1), device=x.device, dtype=torch.long), target[:, :-1]), 1)
                target_pad_2 = torch.cat((torch.zeros((batch_size, 2, 1), device=x.device, dtype=torch.long), target[:,:-2]), 1)
                target_pad_3 = torch.cat((torch.zeros((batch_size, 3, 1), device=x.device, dtype=torch.long), target[:,:-3]), 1)
                return torch.cat((target, target_pad_1, target_pad_2, target_pad_3), -1)
            
            def get_tgt_probs(probs, target):
                r""" 
                args_knn:
                    probs: [B, T, dictionary]
                    target: [B, T]
                Return: [B, T]
                """
                B, T, C = probs.size(0), probs.size(1), probs.size(2)
                one_hot = torch.arange(0, C).to(target.device)[None, None].repeat(B, T, 1) == target[:, :, None]
                return (probs * one_hot.float()).sum(-1)

            def get_4_gram_probs(target_prob):
                r"""
                args_knn:
                    target_prob: [B, T]
                Return: [B, T, 4]
                """
                target_prob = target_prob[:, :, None]
                target_pad_1 = torch.cat((target_prob[:, :1].repeat(1, 1, 1), target_prob[:, :-1]), 1)
                target_pad_2 = torch.cat((target_prob[:, :1].repeat(1, 2, 1), target_prob[:, :-2]), 1)
                target_pad_3 = torch.cat((target_prob[:, :1].repeat(1, 3, 1), target_prob[:, :-3]), 1)
                return torch.cat((target_prob, target_pad_1,  target_pad_2, target_pad_3), -1)

            def get_entropy(probs):
                r"""probs: [B, T, dictionary]"""
                return - (probs * torch.log(probs+1e-7)).sum(-1)

            # calulate probs
            # batch, src_encodings, dec_init_vec = self.init_decode(examples)
            # print(src_encodings)
            res = self.parse_knnmt(eval(examples[0].intent))
            x = res[0].hidden_states
            x = torch.stack(x, dim=0)
            output_probs = res[0].scores
            # for i in output_probs:
            #     print(i)
            output_probs = torch.stack(output_probs, dim=0).unsqueeze(0).squeeze(2)
            # print(output_probs.size())
            target = self.datastore.get_target().unsqueeze(0)
            # print(target.size())
            # print(res[0].actions)
            ids_4_gram = get_4_gram(target) # [B, T, 4]
            target_prob = get_tgt_probs(output_probs, target) # [B, T]
            probs_4_gram = get_4_gram_probs(target_prob) # [B, T, 4]
            entropy = get_entropy(output_probs) # [B, T]
            # process pad
            pad_mask = self.datastore.get_pad_mask()
            keys = select_keys_with_pad_mask(x, pad_mask)
            ids_4_gram = select_keys_with_pad_mask(ids_4_gram, pad_mask)
            probs_4_gram = select_keys_with_pad_mask(probs_4_gram, pad_mask)
            entropy = entropy.masked_select(pad_mask)
            # save infomation to datastore
            self.datastore["keys"].add(keys.half())
            self.datastore["ids_4_gram"].add(ids_4_gram)
            self.datastore["probs_4_gram"].add(probs_4_gram)
            self.datastore["entropy"].add(entropy)

        elif self.args_knn.knn_mode == "train_metak" or self.args_knn.knn_mode == "inference":
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            self.retriever.retrieve(self.datastore.vector_reduct(x), return_list=["vals", "distances"])
        
        # if not features_only:
        #     x = self.output_layer(x)
        return x
    

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1.
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args_knn.knn_mode == "inference" or self.args_knn.knn_mode == "train_metak":
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)



r""" Define some pck knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
# @register_model_architecture("knntranx", "knntranx@tranx")
# def tranx(args_knn):
#     archs.tranx(args_knn)
#
# @register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_iwslt_de_en")
# def transformer_iwslt_de_en(args_knn):
#     archs.transformer_iwslt_de_en(args_knn)
#
# @register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt_en_de")
# def transformer_wmt_en_de(args_knn):
#     archs.base_architecture(args_knn)
#
# # parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
# @register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_vaswani_wmt_en_de_big")
# def transformer_vaswani_wmt_en_de_big(args_knn):
#     archs.transformer_vaswani_wmt_en_de_big(args_knn)
#
# @register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_vaswani_wmt_en_fr_big")
# def transformer_vaswani_wmt_en_fr_big(args_knn):
#     archs.transformer_vaswani_wmt_en_fr_big(args_knn)
#
# @register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt_en_de_big")
# def transformer_wmt_en_de_big(args_knn):
#     archs.transformer_vaswani_wmt_en_de_big(args_knn)
#
# # default parameters used in tensor2tensor implementation
# @register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt_en_de_big_t2t")
# def transformer_wmt_en_de_big_t2t(args_knn):
#     archs.transformer_wmt_en_de_big_t2t(args_knn)
#
# @register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt19_de_en")
# def transformer_wmt19_de_en(args_knn):
#     archs.transformer_wmt19_de_en(args_knn)

    
    

        

