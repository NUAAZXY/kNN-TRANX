:<<! 
[script description]: build a datastore for pck-knn-mt
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
cd $( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..

git lfs install
git clone https://huggingface.co/xyzhang/knn_datastores
mv knn_datastores/datastore datastore
rm -rf knn_datastores

echo 'download down'