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
from collections import OrderedDict
from torch.autograd import Variable
from knnbox.datastore import Datastore, PckDatastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner, AdaptiveCombiner
from knnbox.models.BertranX.model.nl2code import nl2code
from knnbox.models.BertranX.asdl.hypothesis import DecodeHypothesis
from knnbox.models.BertranX.model import nn_utils
import numpy as np
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
            if args_knn.knn_mode == "train_metak" or args_knn.knn_mode == 'inference':
                self.combiner = AdaptiveCombiner(max_k=args_knn.knn_max_k, probability_dim=len(vocab.code)+len(vocab.primitive),
                            k_trainable=(args_knn.knn_k_type=="trainable"),
                            lambda_trainable=(args_knn.knn_lambda_type=="trainable"), lamda_=args_knn.knn_lambda,
                            temperature_trainable=(args_knn.knn_temperature_type=="trainable"), temperature=args_knn.knn_temperature
                                                 # temperature_trainable=(args_knn.knn_temperature_type == "trainable"), temperature=100000

                                                 )
            # elif args_knn.knn_mode == "inference":
            #     self.combiner = AdaptiveCombiner.load(args_knn.knn_combiner_path, args_knn)

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
            res, src = self.parse_knnmt(eval(examples[0].intent))
            x = res[0].hidden_states
            x = torch.stack(x, dim=0)
            output_probs = res[0].scores
            # for i in output_probs:
            #     print(i)
            output_probs = torch.stack(output_probs, dim=0).unsqueeze(0).squeeze(2)
            target = self.datastore.get_target().unsqueeze(0)
            print(output_probs.size(), target.size())
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
            print('--------saving--------')
            self.datastore["keys"].add(keys.half())
            self.datastore["ids_4_gram"].add(ids_4_gram)
            self.datastore["probs_4_gram"].add(probs_4_gram)
            self.datastore["entropy"].add(entropy)

        elif self.args_knn.knn_mode == "train_metak" or self.args_knn.knn_mode == "inference":
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            res, src = self.parse_knnmt_teacher(eval(examples[0].intent), examples)
            self.net_prob = torch.stack(res[0].scores, dim=0)
            x = res[0].hidden_states
            x = torch.stack(x, dim=0).unsqueeze(0)
            self.retriever.retrieve(self.datastore.vector_reduct(x), return_list=["vals", "distances"])
            # self.retriever.retrieve(x, return_list=["vals", "distances"])

        # if not features_only:
        #     x = self.output_layer(x)
        return x, src
    

    def get_normalized_probs(
        self,
        # net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
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
        if self.args_knn.knn_mode == "inference":
            self.retriever.results['distances'] = self.retriever.results['distances'].unsqueeze(0)
            self.retriever.results['vals'] = self.retriever.results['vals'].unsqueeze(0)
        if self.args_knn.knn_mode == "inference" or self.args_knn.knn_mode == "train_metak":
            net_output = self.net_prob
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output.device).squeeze(0)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output, log_probs=False)
            # print(torch.argmax(knn_prob, dim=-1), torch.argmax(net_output, dim=-1))
            # print(torch.max(knn_prob, dim=-1).values, torch.max(net_output, dim=-1).values)
            # print(torch.argmax(combined_prob, dim=-1), torch.max(combined_prob, dim=-1).values)
            # print(net_output, knn_prob)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)

    def parse_gen(self, src_sent):

        primitive_vocab = self.vocab.primitive
        # T = torch.cuda if self.cuda else torch

        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, device=self.device, training=False)

        # Variable(1, src_sent_len, hidden_size * 2)
        src_encodings, last_cell = self.encode(src_sent_var, [len(src_sent)])
        # (1, src_sent_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = self.init_decoder_state(last_cell)

        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]).to(self.device))

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)
        t = 0

        hypotheses = [DecodeHypothesis()]
        completed_hypotheses = []

        while len(completed_hypotheses) < self.args['beam_size'] and t < self.args['len_max']:

            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            if t == 0:

                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).to(self.device).zero_())

            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]

                a_tm1_embeds = []

                for a_tm1 in actions_tm1:
                    # print(a_tm1)
                    # print(type(a_tm1[0]))
                    if a_tm1[0] in self.act_dict:
                        a_tm1_embed = self.action_embed.weight[self.vocab.code[a_tm1[0]]]
                    else:
                        # print(self.vocab.primitive[str(a_tm1[0])])
                        a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[str(a_tm1[0])]]

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds, att_tm1]

                # parents = [hyp.stack[-1][hyp.pointer[-1]] for hyp in hypotheses]

                if self.args['parent_feeding_type'] is True:
                    parent_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                        self.grammar.type2id[parent_type] for parent_type, _, _ in parents]).to(self.device)))
                    inputs.append(parent_type_embeds)

                if self.args['parent_feeding_field'] is True:
                    parent_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                        self.grammar.field2id[parent_field] for _, parent_field, _ in parents]).to(self.device)))
                    inputs.append(parent_field_embeds)

                x = torch.cat(inputs, dim=-1)

            # On calcule l'état décodé prédit précédemment
            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             src_token_mask=None)
            # Variable(batch_size, grammar_size)
            # apply_rule_log_prob = torch.log(F.softmax(self.production_readout(att_t), dim=-1))
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)
            # print(apply_rule_log_prob.size())
            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)
            # print(gen_from_vocab_prob.size())

            if not (self.args['copy']):
                primitive_prob = gen_from_vocab_prob
            else:
                # Variable(batch_size, src_sent_len)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

                # Variable(batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # Variable(batch_size, primitive_vocab_size)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob
            unk_copy_token = []

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_prev_hyp_ids = []
            self.retriever.retrieve(self.datastore.vector_reduct(att_t), return_list=["vals", "distances"])
            self.net_prob = torch.cat([torch.exp(apply_rule_log_prob), primitive_prob], dim=-1)
            final_prob = self.get_normalized_probs(log_probs=False)
            final_prob = final_prob.squeeze(0)
            apply_rule_log_prob = torch.log(final_prob[:, :len(self.vocab.code)])
            primitive_prob = final_prob[:, len(self.vocab.code):]
            for hyp_id, hyp in enumerate(hypotheses):
                action_type = self.get_valid_continuation_types(hyp.rules)
                if action_type == 'ActionRule':
                    mask = self.create_mask_action(hyp.rules)
                    mask_temp = mask
                    if len(mask) == 108:
                        mask_temp.append(False)
                    # print(mask)
                    # print(len(mask))
                    # print(mask.count(True))

                    # print(final_score)
                    productions = [i for (i, bool) in enumerate(mask) if bool]
                    # print('prod', len(productions))
                    for prod_id in productions:
                        prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                        new_hyp_score = hyp.score + prod_score
                        applyrule_new_hyp_prod_ids.append(prod_id)
                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_prev_hyp_ids.append(hyp_id)
                else:
                    # Primitives
                    gentoken_prev_hyp_ids.append(hyp_id)
                    if self.args['copy'] is True:
                        # last_rule = rules.pop(0)
                        for token, token_pos_list in aggregated_primitive_tokens.items():
                            # Get probability token number k (sum to get value)
                            sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0,
                                                         Variable(
                                                             torch.LongTensor(token_pos_list).to(self.device))).sum()

                            # Get global probability copying token number k
                            gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                            if token in primitive_vocab:
                                # For dev_set, always True
                                token_id = primitive_vocab[token]
                                primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob

                            else:
                                # Token unknown
                                unk_copy_token.append({'token': token, 'token_pos_list': token_pos_list,
                                                       'copy_prob': gated_copy_prob.data.item()})

                        if self.args['copy'] is True and len(unk_copy_token) > 0:
                            unk_i = np.array([x['copy_prob'] for x in unk_copy_token]).argmax()
                            token = unk_copy_token[unk_i]['token']
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = unk_copy_token[unk_i]['copy_prob']
                            gentoken_new_hyp_unks.append(token)


            new_hyp_scores = None
            if applyrule_new_hyp_scores:  # si l'hypothese était une action
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores).to(self.device))
            if gentoken_prev_hyp_ids:  # si l(hypothese etait un gentoken
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (
                        hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids,
                                                                         :]).view(-1)

                if new_hyp_scores is None:  # si on a qu'un seul terminal
                    new_hyp_scores = gen_token_new_hyp_scores
                else:
                    new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0),
                                                                   self.args['beam_size'] - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                if new_hyp_pos < len(applyrule_new_hyp_scores):
                    prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                    prev_hyp = hypotheses[prev_hyp_id].copy()
                    prev_hyp.hidden_states.append(att_t[prev_hyp_id])
                    prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                    apply_rule_log_prob_temp = apply_rule_log_prob[prev_hyp_id]
                    apply_rule_log_prob_temp[~torch.tensor(mask_temp)] = -float('inf')
                    apply_rule_log_prob_temp = torch.exp(apply_rule_log_prob_temp)
                    zero_vec = torch.zeros_like(gen_from_vocab_prob[0])
                    final_score = torch.cat([apply_rule_log_prob_temp, zero_vec], dim=-1)
                    prev_hyp.scores.append(final_score)
                    # print([(self.vocab.code.id2word[i], i) for i in range(100)])
                    production = self.vocab.code.id2word[prod_id]
                    action = self.act_dict[production]
                    prev_hyp.actions.append((action.label, prev_hyp.rules[0][1]))
                    prev_hyp.rules = action.apply(prev_hyp.rules, self.primitives_type)
                    if action.rhs == []:
                        action_label = action.label
                        # prev_hyp.pointer = self.shift(prev_hyp.pointer)
                        prev_hyp.pointer, prev_hyp.stack = self.completion(action_label, prev_hyp.pointer,
                                                                           prev_hyp.stack)
                    else:
                        prev_hyp.stack.append([*zip(action.rhs, action.rhs_names, action.iter_flags)])
                        prev_hyp.pointer.append(0)

                    # prev_hyp.scores.append(new_hyp_score - prev_hyp.score)
                    prev_hyp.score = new_hyp_score
                else:
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1)
                    k = (new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(1)

                    prev_hyp_id = gentoken_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id].copy()
                    prev_hyp.hidden_states.append(att_t[prev_hyp_id])
                    last_rule = prev_hyp.rules[0]
                    zero_vec = torch.zeros_like(apply_rule_log_prob[0])
                    final_score = torch.cat([zero_vec, primitive_prob[prev_hyp_id]], dim=-1)
                    prev_hyp.scores.append(final_score)
                    if last_rule[1] == '-' or last_rule[1] == '?':
                        prev_hyp.rules.pop(0)

                        if token_id == primitive_vocab.unk_id:
                            if gentoken_new_hyp_unks:
                                token = self.terminal_type(gentoken_new_hyp_unks[k])
                            else:
                                token = self.terminal_type(primitive_vocab.id2word[primitive_vocab.unk_id])
                        else:
                            token = self.terminal_type(primitive_vocab.id2word[token_id.item()])

                        if token == 'Reduce_primitif':
                            prev_hyp.actions.append((token, '?'))
                        else:
                            prev_hyp.actions.append((token, last_rule[1]))

                        action = token

                        prev_hyp.pointer, prev_hyp.stack = self.completion(token, prev_hyp.pointer, prev_hyp.stack)
                        prev_hyp.score = new_hyp_score

                    else:

                        last_rule = prev_hyp.rules[0]
                        if token_id == primitive_vocab.unk_id:
                            if gentoken_new_hyp_unks:
                                token = self.terminal_type(gentoken_new_hyp_unks[k])
                            else:
                                token = self.terminal_type(primitive_vocab.id2word[primitive_vocab.unk_id])
                        else:
                            token = self.terminal_type(primitive_vocab.id2word[token_id.item()])

                        if token == 'Reduce_primitif':
                            prev_hyp.actions.append((token, '?'))
                            prev_hyp.rules.pop(0)
                        else:
                            prev_hyp.actions.append((token, last_rule[1]))

                        action = token

                        prev_hyp.pointer, prev_hyp.stack = self.completion(token, prev_hyp.pointer, prev_hyp.stack)
                        prev_hyp.score = new_hyp_score
                new_hyp = prev_hyp

                if new_hyp.rules == []:
                    new_hyp.score /= (t + 1)
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                # hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]).to(self.device))
                t += 1
            else:
                break

            # print(t)
        # print(completed_hypotheses[0].actions)
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses
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

    
    

        

