import pickle
import re
import traceback

import nltk
from transformers import BertTokenizer

# nltk.download('punkt')

from knnbox.models.BertranX.dataset.utils import tokenize_for_bleu_eval
from knnbox.models.BertranX.asdl.ast_operation import *

QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")


def preprocess_data_conala(raw_path_conala, act_dict, params):
    """
    data processing from CoNaLa raw token
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['var_0', 'str_0', 'var_1', 'str_1', 'var_2', 'str_2', 'var_3', 'str_3', 'var_4', 'str_4'])

    for file_path, file_type in [(raw_path_conala + 'conala-train.json', 'annotated'),
                                 (raw_path_conala + 'conala-test.json', 'annotated'),
                                 (raw_path_conala + 'conala-mined.jsonl', 'mined')
                                 ]:
        print('file {}'.format(file_path), file=sys.stderr)
        del_list = []
        if file_type == 'annotated':
            dataset = json.load(open(file_path))
        elif file_type == 'mined':
            dataset = []
            with open(file_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))

        print('number of examples before preprocessing', len(dataset))

        for i, example in enumerate(dataset):
            intent = example['intent']
            snippet = example['snippet']

            if file_type == 'annotated':
                rewritten_intent = example['rewritten_intent']
            elif file_type == 'mined':
                rewritten_intent = example['intent']

            if rewritten_intent:
                try:
                    intent_tokens, slot_map = canonicalize_intent(rewritten_intent, params, tokenizer)
                    encoded_reconstr_code = snippet
                except:
                    print('*' * 20, file=sys.stderr)
                    print(i, file=sys.stderr)
                    print(intent, file=sys.stderr)
                    print(snippet, file=sys.stderr)
                    traceback.print_exc()

                    failed = True
            if not intent_tokens:
                canonical_intent, slot_map = canonicalize_intent(intent, params)

            if rewritten_intent is None:
                encoded_reconstr_code = snippet.strip()

            example['intent_tokens'] = intent_tokens

            try:
                encoded_reconstr_code_canonical = canonicalize_code(encoded_reconstr_code, slot_map)
                py_ast = ast.parse(encoded_reconstr_code_canonical)
                py_ast = py_ast.body[0]

                actions, type, field, cardinality = ast2seq(py_ast, act_dict)

                encoded_actions = [
                    action[0].label if isinstance(action[0], GrammarRule) or isinstance(action[0], ReduceAction)
                    else (str(action[0]))
                    for action in actions]

                example['snippet_actions'] = encoded_actions

                encoded_reconstr_actions_1 = [*zip([act_dict[encoded_action] if encoded_action in act_dict
                                                    else act_dict['Reduce'] if encoded_action == 'Reduce_primitif'
                else (terminal_type(encoded_action)) for encoded_action in encoded_actions],
                                                   [action[1] for action in actions])]

                example['type'] = type
                example['field'] = field
                example['cardinality'] = cardinality
                example['slot_map'] = slot_map

                assert len(type) == len(field)

                code = seq2ast(make_iterlists(deque(encoded_reconstr_actions_1)))
                code = astor.to_source(code).rstrip()
                code = decanonicalize_code_conala(code, slot_map)
                snippet_tokens = tokenize_for_bleu_eval(code)
                example['snippet_tokens'] = snippet_tokens

            except:
                del_list.append(i)
                # print(encoded_reconstr_code)
                # print(encoded_actions)
                # print(code)
                # print(astor.to_source(code).rstrip())

        dataset = [data for i, data in enumerate(dataset) if i not in del_list]

        print('number of examples preprocessed', len(dataset))

        pickle.dump(dataset, open(file_path + '.seq2seq', 'wb'))


def canonicalize_intent(intent, params, tokenizer='None'):
    # handle the following special case: quote is `''`
    if params['change_term_name'] == True:
        marked_token_matches = QUOTED_TOKEN_RE.findall(intent)

        slot_map = dict()
        var_id = 0
        str_id = 0
        for match in marked_token_matches:
            quote = match[0]
            value = match[1]
            quoted_value = quote + value + quote

            # try:
            #     # if it's a number, then keep it and leave it to the copy mechanism
            #     float(value)
            #     intent = intent.replace(quoted_value, value)
            #     continue
            # except:
            #     pass

            slot_type = infer_slot_type(quote, value)

            if slot_type == 'var':
                slot_name = 'var_%d' % var_id
                var_id += 1
                slot_type = 'var'
            else:

                slot_name = 'str_%d' % str_id
                str_id += 1
                slot_type = 'str'

            # slot_id = len(slot_map)
            # slot_name = 'slot_%d' % slot_id
            # # make sure slot_name is also unicode
            # slot_name = unicode(slot_name)
            # is_list = is_enumerable_str(value.strip().encode().decode('unicode_escape', 'ignore'))
            # if not is_list:

            intent = intent.replace(quoted_value, slot_name)

            slot_map[slot_name] = {'value': value.strip().encode().decode('unicode_escape', 'ignore'),
                                   'quote': quote,
                                   'type': slot_type}

            # else:
            #     pass
    else:
        slot_map = dict()

    if params['model'] == 'bert':
        intent = tokenizer.tokenize(intent.lower())
        intent = ['[CLS]'] + intent + ['[SEP]']
        return intent, slot_map
    else:
        return nltk.word_tokenize(intent.lower()), slot_map


def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'


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


def canonicalize_code(code, slot_map):
    string2slot = {x['value']: slot_name for slot_name, x in list(slot_map.items())}
    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, string2slot)
    canonical_code = astor.to_source(py_ast).strip()

    # the following code handles the special case that
    # a list/dict/set mentioned in the intent, like
    # Intent: zip two lists `[1, 2]` and `[3, 4]` into a list of two tuples containing elements at the same index in each list
    # Code: zip([1, 2], [3, 4])

    entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if is_enumerable_str(val['value'])]
    if entries_that_are_lists:
        for slot_name in entries_that_are_lists:
            list_repr = slot_map[slot_name]['value']
            # if list_repr[0] == '[' and list_repr[-1] == ']':
            first_token = list_repr[0]  # e.g. `[`
            last_token = list_repr[-1]  # e.g., `]`
            fake_list = first_token + slot_name + last_token
            slot_map[fake_list] = slot_map[slot_name]
            # else:
            #     fake_list = slot_name

            canonical_code = canonical_code.replace(list_repr, fake_list)

    return canonical_code


def decanonicalize_code_conala(code, slot_map):
    for slot_name, slot_val in slot_map.items():
        if is_enumerable_str(slot_name):
            code = code.replace(slot_name, slot_val['value'])

    slot2string = {x[0]: x[1]['value'] for x in list(slot_map.items())}
    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, slot2string)
    raw_code = astor.to_source(py_ast).strip()
    # for slot_name, slot_info in slot_map.items():
    #     raw_code = raw_code.replace(slot_name, slot_info['value'])

    return raw_code


def is_enumerable_str(identifier_value):
    """
    Test if the quoted identifier value is a list
    """

    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in (
        '}', ']', ')')


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

