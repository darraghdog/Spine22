
# Run inference of all images. 
for FOLD in 0 1 2 3 4 # -1

do
    for MODEL_NAME in cfg_dh_seg_02G cfg_dh_seg_04A cfg_dh_seg_04F
    do
   	search_dir="weights/${MODEL_NAME}/fold${FOLD}"
        for WEIGHTS_NAME in "$search_dir"/check*.pth
        do
            echo "Running inference on "$WEIGHTS_NAME
                python train.py -C $MODEL_NAME'_test' --fold $FOLD --pretrained_weights $WEIGHTS_NAME
        done
    done
done

# Aggregate vertebrae predictions and make fracture label
python _make_seg_labels_part2.py

