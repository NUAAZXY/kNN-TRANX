:<<!
[script description]: use pck-mt datastore and adaptive combiner to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
cd $( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/knnbox/models/BertranX/outputs/bert.dataset_conala._word_freq_1.nl_embed_256.action_embed_256.att_size_512.hidden_size_512.epochs_60.dropout_enc_0.dropout_dec_0.3.batch_size16.parent_feeding_type_False.parent_feeding_field_False.change_term_name_True.seed_9876/model.pt
DATA_PATH=$PROJECT_PATH/home/ubuntu/knnmt/knnbox/models/BertranX/dataset/data_conala/test/conala-test.csv
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/knntranx_adaptive/wmine_teacher/
COMBINER_LOAD_DIR=$PROJECT_PATH/save-models/combiner/knntranx_adaptive/wmine_teacher/
MAX_K=8

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate4knntranx_adaptive.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 20 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--max-tokens 2048 \
--scoring sacrebleu \
--tokenizer moses --remove-bpe \
--arch pck_knn_mt@transformer_wmt19_de_en \
--user-dir $PROJECT_PATH/knnbox/models \
--knn-mode inference \
--knn-datastore-path  $DATASTORE_LOAD_PATH \
--knn-max-k $MAX_K \
--knn-combiner-path $COMBINER_LOAD_DIR \
--seed 9876 \
