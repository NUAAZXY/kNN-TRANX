#!/bin/bash
set -e

# Set absolute path
export PYTHONPATH="$PWD/dataset/:$PWD/dataset/data_conala/:$PWD/model/:$PWD/dataset/data_conala/conala-corpus/"

# Get the data
echo "download CoNaLa dataset"
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip -d ./dataset/data_conala
rm -r conala-corpus-v1.1.zip
echo "CoNaLa done"

#echo "download CodeSearchNet dataset"
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
#unzip python.zip -d ./dataset/data_github
#rm -r python.zip
#echo "CodeSearchNet done"

#echo "download APPS dataset"
#wget https://people.eecs.berkeley.edu/\~hendrycks/APPS.tar.gz --no-check-certificate
#tar -xzf APPS.tar.gz -C dataset
#rsync -a dataset/apps_dataset/APPS dataset/data_apps
#rm -r APPS.tar.gz
#rm -r dataset/apps_dataset
#echo "APPS done"
git lfs install
git clone https://huggingface.co/xyzhang/knntranx
git clone https://huggingface.co/xyzhang/knntranx_django
mkdir outputs
mv knntranx outputs/bert.dataset_conala._word_freq_1.nl_embed_256.action_embed_256.att_size_512.hidden_size_512.epochs_60.dropout_enc_0.dropout_dec_0.3.batch_size16.parent_feeding_type_False.parent_feeding_field_False.change_term_name_True.seed_9876
mv knntranx_django outputs/bert.dataset_django._word_freq_1.nl_embed_256.action_embed_256.att_size_512.hidden_size_512.epochs_40.dropout_enc_0.dropout_dec_0.6.batch_size32.parent_feeding_type_False.parent_feeding_field_False.change_term_name_True.seed_2200
# Preprocess data

python get_data.py \
    config/config.yml

