CONFIG_FILE=config/lightning.yaml
ORIG_WEIGHT_FILE=$1
WEIGHT_FILE=$(mktemp $ORIG_WEIGHT_FILE.XXXX)
TMP_FILE=$2

echo $WEIGHT_FILE
cp $ORIG_WEIGHT_FILE $WEIGHT_FILE

# remove temporary file when the program terminates (similar to a `finally` clause)
trap 'rm $WEIGHT_FILE' EXIT

cd evar || exit 1

CUDA_VISIBLE_DEVICES=0 python lineareval.py $CONFIG_FILE cremad batch_size=16,weight_file=$WEIGHT_FILE --verbose=1
CUDA_VISIBLE_DEVICES=0 python lineareval.py $CONFIG_FILE gtzan batch_size=16,weight_file=$WEIGHT_FILE --verbose=1
CUDA_VISIBLE_DEVICES=0 python lineareval.py $CONFIG_FILE spcv2 batch_size=16,weight_file=$WEIGHT_FILE --verbose=1
CUDA_VISIBLE_DEVICES=0 python lineareval.py $CONFIG_FILE esc50 batch_size=16,weight_file=$WEIGHT_FILE --verbose=1

python summarize.py $WEIGHT_FILE $TMP_FILE
