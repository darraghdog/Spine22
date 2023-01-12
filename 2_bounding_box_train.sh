for _ in 1 2 3
do
    for FOLD in 0 1 2 3 4 -1
    do
        echo "Train fold" $FOLD
        python train.py -C cfg_loc_dh_01B --fold $FOLD
    done
done
