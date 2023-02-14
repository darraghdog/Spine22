# PART 1
# Train the initial model on slice level vertebrae and fracture labels
# Runtime approximately 10 hours for each fold - so total 100 hours
for FOLD in -1 -1 -1 -1 # 0
do
    python train.py -C cfg_dh_fracseq_04F_crop --fold $FOLD
done
```
for FOLD in 0 -1 -1 -1
do
    python train.py -C cfg_dh_fracseq_04G_crop --fold $FOLD
done
```
