# coding=utf-8

from __future__ import print_function

import re

import nltk
import pandas as pd
from transformers import BertTokenizer
from knnbox.models.BertranX.asdl.ast_operation import *
from knnbox.models.BertranX.asdl.grammar import *
from knnbox.models.BertranX.dataset.utils import tokenize_for_bleu_eval

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")


def replace_string_ast_nodes(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]
                    # Python 3
                    # if isinstance(slot_name, unicode):
                    #     try: slot_name = slot_name.encode('ascii')
                    #     except: pass

                    setattr(node, k, slot_name)


def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]
                    # Python 3
                    # if isinstance(slot_name, unicode):
                    #     try: slot_name = slot_name.encode('ascii')
                    #     except: pass

                    setattr(node, k, slot_name)


def enhance_vocab_bert(nl):
    tokens = ['var_0', 'str_0', 'var_1', 'str_1', 'var_2', 'str_2', 'var_3', 'str_3', 'var_4', 'str_4', 'str', 'inst',
              'elt', 'args', 'pos', 'rhs', 'kwargs', 'X', 'unbound']
    # tokens = ['var_0', 'str_0', 'var_1', 'str_1', 'var_2', 'str_2', 'var_3', 'str_3', 'var_4', 'str_4', 'django']
    values = []
    for idx, src_query in enumerate(open(nl)):
        tokens += re.findall(r'[a-zA-Z.]*_\w+', src_query)
        tokens += re.findall(r'[a-zA-Z.]*\.\w+', src_query)
        tokens += re.findall(r'[A-Z.]+\w+', src_query)
    tokens = list(filter(('').__ne__, tokens))
    return tokens


class Django(object):
    @staticmethod
    def canonicalize_code(code):
        if p_elif.match(code):
            code = 'if True: pass\n' + code

        if p_else.match(code):
            code = 'if True: pass\n' + code

        if p_try.match(code):
            code = code + 'pass\nexcept: pass'
        elif p_except.match(code):
            code = 'try: pass\n' + code
        elif p_finally.match(code):
            code = 'try: pass\n' + code

        if p_decorator.match(code):
            code = code + '\ndef dummy(): pass'

        if code[-1] == ':':
            code = code + 'pass'

        return code

    @staticmethod
    def canonicalize_str_nodes(py_ast, identifier2slot):
        for node in ast.walk(py_ast):
            for k, v in list(vars(node).items()):
                if k in ('lineno', 'col_offset', 'ctx'):
                    continue
                # Python 3
                if isinstance(v, str):
                    if v in identifier2slot:
                        slot_name = identifier2slot[v]
                        # Python 3
                        # if isinstance(slot_name, unicode):
                        #     try: slot_name = slot_name.encode('ascii')
                        #     except: pass

                        setattr(node, k, slot_name)

    @staticmethod
    def decanonicalize_code_django(code, slot_str):
        for slot_name, slot_val in slot_str.items():
            code = code.replace(slot_val, slot_name)

        #slot2string = {x[0]: x[1] for x in list(slot_str.items())}
        # py_ast = ast.parse(code)
        # replace_identifiers_in_ast(py_ast, slot_str)
        # raw_code = astor.to_source(py_ast).strip()
        # print(raw_code)
        # for slot_name, slot_info in slot_map.items():
        #     raw_code = raw_code.replace(slot_name, slot_info['value'])

        return code

    @staticmethod
    def canonicalize_query(query, params, tokenizer=None, variables=[]):
        """
        canonicalize the query, replace strings to a special place holder
        """
        if params['change_term_name'] == True:
            str_count = 0
            str_map = dict()

            matches = QUOTED_STRING_RE.findall(query)

            # de-duplicate
            cur_replaced_strs = set()
            for match in matches:
                # If one or more groups are present in the pattern,
                # it returns a list of groups
                quote = match[0]
                str_literal = match[1]
                quoted_str_literal = quote + str_literal + quote

                if str_literal in cur_replaced_strs:
                    # replace the string with new quote with slot id
                    query = query.replace(quoted_str_literal, str_map[str_literal])
                    continue

                # FIXME: substitute the ' % s ' with
                if str_literal in ['%s']:
                    continue

                str_repr = 'str_%d' % str_count
                str_map[str_literal] = str_repr

                query = query.replace(quoted_str_literal, str_repr)

                str_count += 1
                cur_replaced_strs.add(str_literal)

            for var_count, variable in enumerate(variables):
                var_repr = 'var_%d' % var_count
                str_map[variable] = var_repr

                query = query.replace(variable, var_repr)
        else:
            str_map = dict()

        # tokenize
        if params['model'] == 'bert':
            query_tokens = tokenizer.tokenize(query.lower())
            query_tokens = ['[CLS]'] + query_tokens + ['[SEP]']
        else:
            query_tokens = nltk.word_tokenize(query)

        new_query_tokens = []
        # break up function calls like foo.bar.func
        for token in query_tokens:
            new_query_tokens.append(token)
            i = token.find('.')
            if 0 <= i < len(token) - 1:
                new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
                new_query_tokens.extend(new_tokens)
        # query = ' '.join(query_tokens)
        query = ' '.join(new_query_tokens)
        query = query.replace('\' % s \'', '%s').replace('\" %s \"', '%s')

        return query, str_map

    @staticmethod
    def canonicalize_example(query, code, params, tokenizer, act_dict):

        canonical_code = Django.canonicalize_code(code)
        py_ast = ast.parse(canonical_code).body[0]
        actions, type, field, cardinality = ast2seq(py_ast, act_dict)
        variables = [str(action[0]) for action in actions if
                     not isinstance(action[0], GrammarRule) and not isinstance(action[0], ReduceAction) and not isinstance(action[0], bool) and not isinstance(action[0], int)]
        variables = sorted(variables, key=len, reverse=True)
        # variables = [variable for variable in variables if
        #              variable not in ['0', '1', '2', '3', '4', '5', '6', '7', '8']]

        canonical_query, str_map = Django.canonicalize_query(query, params, tokenizer, variables)
        query_tokens = canonical_query.split(' ')

        canonical_code = Django.canonicalize_code(code)
        ast_tree = ast.parse(canonical_code)

        ground_truth = astor.to_source(ast_tree)

        Django.canonicalize_str_nodes(ast_tree, str_map)
        canonical_code = astor.to_source(ast_tree)

        return query_tokens, ground_truth, canonical_code, str_map

    @staticmethod
    def parse_django_dataset(annot_file, code_file, params, act_dict, max_query_len=70):
        i = 0
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        # a = enhance_vocab_bert(annot_file)
        # tokenizer.add_tokens(a)
        tokenizer.add_tokens(
            ['var_0', 'str_0', 'var_1', 'str_1', 'var_2', 'str_2', 'var_3', 'str_3', 'var_4', 'str_4', 'var_5', 'str_5',
             'var_6', 'str_6', 'None', 'True', 'False'])
        tokenizer.unique_no_split_tokens.sort(key=lambda x: -len(x))
        # print(tokenizer.get_vocab())

        loaded_examples = []

        print('number of example before preprocessing = 18805', file=sys.stderr)

        for idx, (src_query, tgt_code) in enumerate(zip(open(annot_file, encoding='ascii'), open(code_file))):
            try:
                src_query = src_query.strip()
                tgt_code = tgt_code.strip()

                src_query_tokens, ground_truth, tgt_canonical_code, str_map = Django.canonicalize_example(src_query,
                                                                                                          tgt_code,
                                                                                                          params,
                                                                                                          tokenizer,
                                                                                                          act_dict)
                py_ast = ast.parse(tgt_canonical_code).body[0]

                actions, type, field, cardinality = ast2seq(py_ast, act_dict)

                encoded_actions = [
                    action[0].label if isinstance(action[0], GrammarRule) or isinstance(action[0], ReduceAction)
                    else (str(action[0]))
                    for action in actions]

                encoded_reconstr_actions_1 = [*zip([act_dict[encoded_action] if encoded_action in act_dict
                                                    else act_dict['Reduce'] if encoded_action == 'Reduce_primitif'
                else (terminal_type(encoded_action)) for encoded_action in encoded_actions],
                                                   [action[1] for action in actions])]

                code = seq2ast(make_iterlists(deque(encoded_reconstr_actions_1)))
                code = astor.to_source(code).rstrip()

                if len(encoded_actions) > 150 or len(src_query_tokens) > 150 or len(
                        tokenize_for_bleu_eval(ground_truth)) > 150:
                    i += 1
                else:
                    loaded_examples.append({'intent': src_query_tokens,
                                            'snippet_tokens': tokenize_for_bleu_eval(ground_truth.strip()),
                                            'snippet_actions': encoded_actions,
                                            'slot_map': str_map})
            except:
                i += 1
        print(len(loaded_examples))
        print('first pass, processed %d' % (idx - i), file=sys.stderr)
        test_examples = loaded_examples[-1805::]
        test_examples = pd.DataFrame(test_examples)
        dev_examples = loaded_examples[-2805:-1805]
        dev_examples = pd.DataFrame(dev_examples)
        train_examples = loaded_examples[:-2805]
        train_examples = pd.DataFrame(train_examples)

        return train_examples, dev_examples, test_examples

    @staticmethod
    def process_django_dataset(params, act_dict):
        annot_file = 'dataset/data_django/all.anno'
        code_file = 'dataset/data_django/all.code'

        (train, dev, test) = Django.parse_django_dataset(annot_file, code_file, params, act_dict)

        train.to_csv('dataset/data_django/train.csv', index=False)
        dev.to_csv('dataset/data_django/dev.csv', index=False)
        test.to_csv('dataset/data_django/test.csv', index=False)


def terminal_type(terminal):
    try:
        a = float(terminal)
    except (TypeError, ValueError, OverflowError):
        try:
            a = ast.literal_eval(terminal)
            return a
        except:
            return str(terminal)
    else:
        try:
            b = int(a)
        except (TypeError, ValueError, OverflowError):
            return a
        else:
            return b


if __name__ == '__main__':
    Django.process_django_dataset()
