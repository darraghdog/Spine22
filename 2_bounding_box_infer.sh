
# Run bounding box inference of all images. 
for FOLD in 0 1 2 3 4 # -1
do
    search_dir='weights/cfg_loc_dh_01B/fold'
    search_dir+=$FOLD
    for WEIGHTS_NAME in "$search_dir"/*
    do
        echo "Running inference on "$WEIGHTS_NAME
        python train.py -C cfg_loc_dh_01B_test --fold $FOLD --pretrained_weights $WEIGHTS_NAME
    done
done

