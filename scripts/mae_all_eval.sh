CONFIG_FILE=config/audiomae.yaml
WEIGHT_FILE=external/audiomae/pretrained.pth
TMP_FILE=$2

cd evar || exit 1

CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE cremad batch_size=16,weight_file=$WEIGHT_FILE
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE gtzan batch_size=16,weight_file=$WEIGHT_FILE
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE spcv2 batch_size=64,weight_file=$WEIGHT_FILE
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE esc50 batch_size=64,weight_file=$WEIGHT_FILE
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE us8k batch_size=64,weight_file=$WEIGHT_FILE
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE vc1 batch_size=64,weight_file=$WEIGHT_FILE
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE nsynth batch_size=64,weight_file=$WEIGHT_FILE
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py $CONFIG_FILE surge batch_size=64,weight_file=$WEIGHT_FILE

echo "Summarizing results..."
python summarize.py $WEIGHT_FILE $TMP_FILE
