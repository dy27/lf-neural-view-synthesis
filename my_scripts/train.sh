TRAIN_PATH=~/datasets/seq82_full/transforms_single_train.json
TEST_PATH=~/datasets/seq82_full/transforms_test.json
N_STEPS=1000
SAVE_SNAPSHOT=~/datasets/seq82_full/single_model.msgpack

python scripts/run.py \
    --mode nerf \
    --scene $TRAIN_PATH \
    --test_transforms $TEST_PATH \
    --n_steps $N_STEPS \
    --save_snapshot $SAVE_SNAPSHOT