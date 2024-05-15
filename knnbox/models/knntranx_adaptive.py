from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
import torch.nn.functional as F
from knnbox.models.BertranX.dataset.dataset import Batch
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

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# @register_model("knntranx")
class KNNTRANX_adaptive(nl2code):
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
            if "datastore_act" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore_act"] = Datastore(
                        path=args_knn.knn_datastore_path+'/act',
                    )
                global_vars()["datastore_pri"] = Datastore(
                    path=args_knn.knn_datastore_path+'/pri',
                )
            self.datastore_act = global_vars()["datastore_act"]
            self.datastore_pri = global_vars()["datastore_pri"]

        else:
            self.datastore_act = Datastore.load(args_knn.knn_datastore_path+'/act', load_list=["vals"])
            self.datastore_pri = Datastore.load(args_knn.knn_datastore_path+'/pri', load_list=["vals"])

            self.datastore_act.load_faiss_index("keys")
            self.datastore_pri.load_faiss_index("keys")

            # self.retriever_act = Retriever(datastore=self.datastore_act, k=4)
            # self.retriever_pri = Retriever(datastore=self.datastore_pri, k=16)

            # if args_knn.knn_mode == "train_metak":
            #     self.combiner_pri = AdaptiveCombiner(max_k=16, probability_dim=len(vocab.primitive),
            #                 k_trainable=(args_knn.knn_k_type=="trainable"),
            #                 lambda_trainable=(args_knn.knn_lambda_type=="trainable"), lambda_=args_knn.knn_lambda,
            #                 temperature_trainable=(args_knn.knn_temperature_type=="trainable"), temperature=args_knn.knn_temperature
            #                                      )
            #     self.combiner_act = AdaptiveCombiner(max_k=4,
            #                                          probability_dim=len(vocab.code),
            #                                          k_trainable=(args_knn.knn_k_type == "trainable"),
            #                                          lambda_trainable=(args_knn.knn_lambda_type == "trainable"),
            #                                          lambda_=args_knn.knn_lambda,
            #                                          temperature_trainable=(
            #                                                      args_knn.knn_temperature_type == "trainable"),
            #                                          temperature=args_knn.knn_temperature
            #                                          )
            # else:
            #     self.combiner_pri = AdaptiveCombiner.load(args_knn.knn_combiner_path+'/pri', len(vocab.primitive), 16)
            #     self.combiner_act = AdaptiveCombiner.load(args_knn.knn_combiner_path+'/act', len(vocab.code), 4

            self.retriever_act = Retriever(datastore=self.datastore_act, k=4)
            self.retriever_pri = Retriever(datastore=self.datastore_pri, k=64)
            if args_knn.knn_mode == "train_metak":

                self.combiner_pri = AdaptiveCombiner(max_k=64, probability_dim=len(vocab.primitive),
                            k_trainable=(args_knn.knn_k_type=="trainable"),
                            lambda_trainable=(args_knn.knn_lambda_type=="trainable"), lambda_=args_knn.knn_lambda,
                            temperature_trainable=(args_knn.knn_temperature_type=="trainable"), temperature=args_knn.knn_temperature
                                                 )
                self.combiner_act = AdaptiveCombiner(max_k=4,
                                                     probability_dim=len(vocab.code),
                                                     k_trainable=(args_knn.knn_k_type == "trainable"),
                                                     lambda_trainable=(args_knn.knn_lambda_type == "trainable"),
                                                     lambda_=args_knn.knn_lambda,
                                                     temperature_trainable=(
                                                                 args_knn.knn_temperature_type == "trainable"),
                                                     temperature=args_knn.knn_temperature
                                                     )
            else:
                self.combiner_pri = AdaptiveCombiner.load(args_knn.knn_combiner_path+'/pri', len(vocab.primitive), 64)
                self.combiner_act = AdaptiveCombiner.load(args_knn.knn_combiner_path+'/act', len(vocab.code), 4)



    def forward(
        self,
        examples,
        **kwargs
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """

        if self.args_knn.knn_mode == "build_datastore":
            # calulate probs
            # batch, src_encodings, dec_init_vec = self.init_decode(examples)
            # print(src_encodings)

            res, src = self.parse_knnmt_teacher(eval(examples[0].intent), examples)
            actions = []
            x = res[0].hidden_states
            output_probs = res[0].scores
            mark = 0
            for i in res:
                action = []
                for j in i.actions:
                    action.append(j[0])
                if action == eval(examples[0].snippet_actions):
                    # print(action, examples[0].snippet_actions)
                    true_pred = i
                    x = true_pred.hidden_states
                    output_probs = true_pred.scores
                    # print(True)
                    mark = 1
                    break
            # this is evil!!
            assert mark == 1,'Cannnot generate correct answer'
            x = torch.stack(x, dim=0)
            act_idx = kwargs['act_idx']
            pri_idx = kwargs['pri_idx']
            epoch = kwargs['step']
            x_act = torch.stack([x[i] for i in act_idx], dim=0)
            x_pri = torch.stack([x[i] for i in pri_idx], dim=0)
            if epoch < 717400:
                keys_act = select_keys_with_pad_mask(x_act, self.datastore_act.get_pad_mask().to('cuda:0'))
                self.datastore_act["keys"].add(keys_act.half())
            keys_pri = select_keys_with_pad_mask(x_pri, self.datastore_pri.get_pad_mask().to('cuda:0'))
            self.datastore_pri["keys"].add(keys_pri.half())


        elif self.args_knn.knn_mode == "train_metak" or self.args_knn.knn_mode == "inference":
            # query with x (x needn't to be half precision),
            # save retrieved `vals` and `distances`
            # res, src = self.parse_knnmt_teacher(eval(examples[0].intent), examples)
            res, src = self.parse_knnmt_teacher(eval(examples[0].intent), examples)
            self.net_prob = torch.stack(res[0].scores, dim=0)
            x = res[0].hidden_states
            x = torch.stack(x, dim=0).unsqueeze(0)
            self.hidden = x
            self.retriever_act.retrieve(x, return_list=["vals", "distances"])
            self.retriever_pri.retrieve(x, return_list=["vals", "distances"])

            # self.retriever.retrieve(x, return_list=["vals", "distances"])

        # if not features_only:
        #     x = self.output_layer(x)
        return x, src


    def get_normalized_probs(
        self,
        idx,
        # net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        act=True
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1.
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args_knn.knn_mode == "inference":
            if act == True:
                self.retriever_act.results['distances'] = self.retriever_act.results['distances'].unsqueeze(0).unsqueeze(1)
                self.retriever_act.results['vals'] = self.retriever_act.results['vals'].unsqueeze(0).unsqueeze(1)
            else:
                self.retriever_pri.results['distances'] = self.retriever_pri.results['distances'].unsqueeze(0).unsqueeze(1)
                self.retriever_pri.results['vals'] = self.retriever_pri.results['vals'].unsqueeze(0).unsqueeze(1)
        if self.args_knn.knn_mode == "inference":
            # print(self.net_prob.size(),idx)
            net_output = self.net_prob[idx].unsqueeze(0)
            # print(net_output.size())
            combined_prob = {}
            if act == True:
                net_act = net_output[:,:len(self.vocab.code)]
                knn_prob_act = self.combiner_act.get_knn_prob(**self.retriever_act.results, device=net_output.device).squeeze(0)
                # print(self.retriever_act.results)
                # print(knn_prob_act)
            # print(net_act.size(), knn_prob_act.size())

            # 0.1 0.035
                combined_prob_act, _ = self.combiner_act.get_combined_prob(knn_prob_act, net_act, None, log_probs=log_probs)
            # print(net_output, knn_prob)
                combined_prob['act'] = combined_prob_act
                # combined_prob['act'] = net_act

            else:
                net_pri = net_output[:, len(self.vocab.code):]
                knn_prob_pri = self.combiner_pri.get_knn_prob(**self.retriever_pri.results, device=net_output.device).squeeze(0)
                combined_prob_pri, _ = self.combiner_pri.get_combined_prob(knn_prob_pri, net_pri, None, log_probs=log_probs)
                combined_prob['pri'] = combined_prob_pri
                # combined_prob['pri'] = net_pri

            return combined_prob
        elif self.args_knn.knn_mode == "train_metak":
            net_output = self.net_prob
            # print(net_output.size())
            combined_prob = {}
            net_act = net_output[:, :len(self.vocab.code)]
            knn_prob_act = self.combiner_act.get_knn_prob(**self.retriever_act.results,
                                                          device=net_output.device).squeeze(0)
            combined_prob_act, _ = self.combiner_act.get_combined_prob(knn_prob_act, net_act, None,
                                                                       log_probs=log_probs)
            combined_prob['act'] = combined_prob_act
            net_pri = net_output[:, len(self.vocab.code):]
            knn_prob_pri = self.combiner_pri.get_knn_prob(**self.retriever_pri.results,
                                                          device=net_output.device).squeeze(0)
            combined_prob_pri, _ = self.combiner_pri.get_combined_prob(knn_prob_pri, net_pri, None,
                                                                       log_probs=log_probs)
            combined_prob['pri'] = combined_prob_pri

            return combined_prob







    def parse_gen(self, src_sent):
        check = False
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

            # print(primitive_vocab.id2word[100])
            # print(self.vocab.code.id2word[100])

            # self.retriever_act.retrieve(att_t, return_list=["vals", "distances"])
            # self.retriever_pri.retrieve(att_t, return_list=["vals", "distances"], prod=None)
            #
            self.net_prob = torch.cat([torch.exp(apply_rule_log_prob), primitive_prob], dim=-1)
            # print(F.softmax(self.production_readout(att_t), dim=-1))
            # print(torch.exp(apply_rule_log_prob))
            # assert F.softmax(self.production_readout(att_t), dim=-1).equal(torch.exp(apply_rule_log_prob))
            # print(self.retriever.results['vals'],self.retriever.results['distances'])

            # final_prob = self.get_normalized_probs(log_probs=False)

            # final_prob = torch.softmax(final_prob, dim=-1)
            # final_prob = self.combiner.get_knn_prob(**self.retriever.results, device='cuda:0').squeeze(0)
            # print(final_prob.size())
            # final_prob = self.net_prob
            # final_prob = final_prob.squeeze(0)

            # apply_rule_log_prob = torch.log(final_prob['act']).squeeze(0)
            # primitive_prob = final_prob['pri'].squeeze(0)
            # primitive_prob = self.net_prob[:, len(self.vocab.code):]
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
                    self.retriever_act.retrieve(att_t[hyp_id], return_list=["vals", "distances"], prod=productions)
                    # self.retriever_act.retrieve(att_t[hyp_id], return_list=["vals", "distances"], prod=None)
                    apply_rule_log_prob_ = self.get_normalized_probs(hyp_id, log_probs=True, act=True)['act'].squeeze(0)
                    # print(1,apply_rule_log_prob[hyp_id])
                    # print(2,apply_rule_log_prob_)
                    # print(apply_rule_log_prob)
                    # print(apply_rule_log_prob_.size())

                    # print(1,apply_rule_log_prob[hyp_id])
                    # print(2,apply_rule_log_prob_)


                    apply_rule_log_prob[hyp_id] = apply_rule_log_prob_
                    for prod_id in productions:
                        # prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                        prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                        new_hyp_score = hyp.score + prod_score
                        applyrule_new_hyp_prod_ids.append(prod_id)
                        # print(hyp.score, new_hyp_score)
                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_prev_hyp_ids.append(hyp_id)
                else:
                    # Primitives
                    self.retriever_pri.retrieve(att_t[hyp_id], return_list=["vals", "distances"], prod=None)
                    primitive_prob_ = self.get_normalized_probs(hyp_id, log_probs=False, act=False)['pri'].squeeze(0)
                    # print(primitive_prob_.size())

                    primitive_prob[hyp_id] = primitive_prob_

                    # print(primitive_prob.size())
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
                                # primitive_prob[0, token_id] = primitive_prob[0, token_id] + gated_copy_prob

                            else:
                                # Token unknown
                                unk_copy_token.append({'token': token, 'token_pos_list': token_pos_list,
                                                       'copy_prob': gated_copy_prob.data.item()})

                        if self.args['copy'] is True and len(unk_copy_token) > 0:
                            unk_i = np.array([x['copy_prob'] for x in unk_copy_token]).argmax()
                            token = unk_copy_token[unk_i]['token']
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = unk_copy_token[unk_i]['copy_prob']
                            # primitive_prob[0, primitive_vocab.unk_id] = unk_copy_token[unk_i]['copy_prob']
                            gentoken_new_hyp_unks.append(token)

            # print(apply_rule_log_prob)

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
                    # print(1,new_hyp_scores.size(), gen_token_new_hyp_scores.size())
                    new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])
                    # print(new_hyp_scores.size())
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
                    # print(new_hyp_score)
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
                # print(t, action)
                if new_hyp.rules == []:
                    # new_hyp.score /= (t + 1)
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
        self.length_penalty(completed_hypotheses)
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses



    def length_penalty(self, hyps):
        for hyp in hyps:
            hyp.score /= (len(hyp.actions) + 1) ** 1.2

    def parse_knnmt_teacher_acc(self, src_sent, examples):
        batch = Batch(examples, self.args, self.act_dict, self.vocab, self.device, self.grammar)
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

        while len(completed_hypotheses) < self.args['beam_size'] and t < self.args['len_max'] and t < len(
                eval(examples[0].snippet_actions)):

            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            if t == 0:
                a_tm_tobepred = eval(examples[0].snippet_actions)[t]
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).to(self.device).zero_())

            else:
                a_tm1 = eval(examples[0].snippet_actions)[t - 1]
                a_tm_tobepred = eval(examples[0].snippet_actions)[t]
                a_tm1_embeds = []

                if a_tm1 in self.act_dict:
                    a_tm1_embed = self.action_embed.weight[self.vocab.code[a_tm1]]
                else:
                    # print(self.vocab.primitive[str(a_tm1[0])])
                    a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[str(a_tm1)]]

                a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                # print(a_tm1_embeds.size(), att_tm1.size())
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
            if a_tm_tobepred in self.act_dict:
                next_pred_token_id = batch.apply_rule_idx_matrix[t].squeeze()
                nn_pred_id = torch.argmax(apply_rule_log_prob, dim=-1)
                self.retriever_act.retrieve(att_t[0].unsqueeze(0).unsqueeze(1), return_list=["vals", "distances"], prod=None)
                knn_prob_act = self.combiner_act.get_knn_prob(**self.retriever_act.results,
                                                              device='cuda:0').squeeze(0)
                knn_pred_id = torch.argmax(knn_prob_act, dim=-1)
                combine_prob, extra = self.combiner_act.get_combined_prob(knn_prob_act, torch.exp(apply_rule_log_prob), None, log_probs=False)
                print(nn_pred_id, knn_pred_id, next_pred_token_id, extra['lambda'])
            else:
                next_pred_token_id = batch.primitive_idx_matrix[t].squeeze()
                nn_pred_id = torch.argmax(gen_from_vocab_prob)
                self.retriever_pri.retrieve(att_t[0].unsqueeze(0).unsqueeze(1), return_list=["vals", "distances"], prod=None)
                knn_prob_pri = self.combiner_pri.get_knn_prob(**self.retriever_pri.results,
                                                              device='cuda:0').squeeze(0)
                knn_pred_id = torch.argmax(knn_prob_pri, dim=-1)
                combine_prob, extra = self.combiner_pri.get_combined_prob(knn_prob_pri, knn_prob_pri, None, log_probs=False)

                print(nn_pred_id, knn_pred_id, next_pred_token_id, extra['lambda'])

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
                                                                   1 - len(completed_hypotheses)))

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
                    # production = self.vocab.code.id2word[prod_id]
                    production = eval(examples[0].snippet_actions)[t]
                    action = self.act_dict[production]
                    # prev_hyp.actions.append((action.label, prev_hyp.rules[0][1]))
                    prev_hyp.actions.append((eval(examples[0].snippet_actions)[t], prev_hyp.rules[0][1]))
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
                                # token = self.terminal_type(gentoken_new_hyp_unks[k])
                                token = self.terminal_type(eval(examples[0].snippet_actions)[t])
                            else:
                                # token = self.terminal_type(primitive_vocab.id2word[primitive_vocab.unk_id])
                                token = self.terminal_type(eval(examples[0].snippet_actions)[t])
                        else:
                            # token = self.terminal_type(primitive_vocab.id2word[token_id.item()])
                            token = self.terminal_type(eval(examples[0].snippet_actions)[t])
                        if token == 'Reduce_primitif':
                            # prev_hyp.actions.append((token, '?'))
                            prev_hyp.actions.append((eval(examples[0].snippet_actions)[t], '?'))
                        else:
                            # prev_hyp.actions.append((token, last_rule[1]))
                            prev_hyp.actions.append((eval(examples[0].snippet_actions)[t], last_rule[1]))
                        action = token

                        prev_hyp.pointer, prev_hyp.stack = self.completion(token, prev_hyp.pointer, prev_hyp.stack)
                        prev_hyp.score = new_hyp_score

                    else:

                        last_rule = prev_hyp.rules[0]
                        if token_id == primitive_vocab.unk_id:
                            if gentoken_new_hyp_unks:
                                # token = self.terminal_type(gentoken_new_hyp_unks[k])
                                token = self.terminal_type(eval(examples[0].snippet_actions)[t])
                            else:
                                # token = self.terminal_type(primitive_vocab.id2word[primitive_vocab.unk_id])
                                token = self.terminal_type(eval(examples[0].snippet_actions)[t])
                        else:
                            # token = self.terminal_type(primitive_vocab.id2word[token_id.item()])
                            token = self.terminal_type(eval(examples[0].snippet_actions)[t])

                        if token == 'Reduce_primitif':
                            # prev_hyp.actions.append((token, '?'))
                            prev_hyp.actions.append((eval(examples[0].snippet_actions)[t], '?'))
                            prev_hyp.rules.pop(0)
                        else:
                            # prev_hyp.actions.append((token, last_rule[1]))
                            prev_hyp.actions.append((eval(examples[0].snippet_actions)[t], last_rule[1]))
                        action = token

                        prev_hyp.pointer, prev_hyp.stack = self.completion(token, prev_hyp.pointer, prev_hyp.stack)
                        prev_hyp.score = new_hyp_score
                new_hyp = prev_hyp
                # print(prev_hyp.actions[-1], eval(examples[0].snippet_actions)[t])
                if new_hyp.rules == [] or t == len(eval(examples[0].snippet_actions)) - 1:
                    # if t == len(eval(examples[0].snippet_actions))-1:
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

        return completed_hypotheses, src_encodings




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






