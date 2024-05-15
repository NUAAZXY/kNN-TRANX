:<<! 
[script description]: build a datastore for pck-knn-mt
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
cd $( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/models/bert.dataset_conala._word_freq_1.nl_embed_256.action_embed_256.att_size_512.hidden_size_512.epochs_60.dropout_enc_0.dropout_dec_0.3.batch_size16.parent_feeding_type_False.parent_feeding_field_False.change_term_name_True.seed_9876/model.pt
DATA_PATH=$PROJECT_PATH/models/BertranX/dataset/conala-corpus/conala-train.csv
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/knntranx_adaptive/wmine_teacher

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--dataset-impl mmap \
--valid-subset train \
--skip-invalid-size-inputs-valid-test \
--max-tokens 4096 \
--bpe fastbpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch pck_knn_mt@transformer_wmt19_de_en \
--knn-mode build_datastore \
--knn-datastore-path $DATASTORE_SAVE_PATH \
