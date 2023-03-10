# PART 2
# LOAD the weights of previous model and train whole study on fracture/vertebrae labels 
# Run bounding box inference of all images. 
# Runtime approximately 1 hour per fold, so total 10 hours.
for FOLD in -1
do
    for MODEL_NAME in cfg_dh_fracseq_04F_crop cfg_dh_fracseq_04G_crop
	do
            search_dir="weights/${MODEL_NAME}/fold" 
            search_dir+=$FOLD
            for WEIGHTS_NAME in "$search_dir"/check*
            do
                echo "Running inference on "$WEIGHTS_NAME
                python train.py -C "${MODEL_NAME}_gx1" --fold $FOLD --pretrained_weights $WEIGHTS_NAME
            done
        done
done

