#!/bin/sh
# GPU usage
GPU_ID=1
WORKER_NUMBER=8

# basic set
# DATASET=foodAllmt10

#DATASET=foodexclYIHmt10
#DATASET=foodexclUTownmt10
#DATASET=foodexclArtsmt10
#DATASET=foodexclTechMixedVegmt10
DATASET=foodexclTechChickenmt10

#DATASET=foodexclUTownmt10_fineUTownfew5
#DATASET=food_meta_Arts_train
#DATASET=foodEconomicBeeHoon

#DATASET=foodexclTechMixedVegmt10
#DATASET=foodAllmt10

NET=foodres50attention #foodres50 #{foodres50, res101, vgg16}
#NET=foodres50_hierarchy_casecade_add_prob_0.5 #_casecade #{foodres50, res101, vgg16 , foodres50_hierarchy foodres50attention, foodres502fc, foodres50_hierarchy_casecade}

SESSION=51
FIXED_LAYER=4
PRETRAIN=true
WEIGHT_FILE=imagenet #{ prefood, imagenet } only for res50
MAXEPOCHS=18
SAVE_EPOCH=3

# optimizer setting
OPTIMIZER=sgd
LEARNING_RATE=0.001
DECAY_STEP=10
IS_WARMING_UP=false
WARMING_UP_LR=0.0000001
BATCH_SIZE=1

# resume from
RESUME= # null is for false
RESUME_OPT= # null for false
RESUME_SESS_EPOCH= #null for false
CHECKSESSION=444
CHECKEPOCH=35
CHECKPOINT=11407


# writing the experiment detail to file
filename=./Experiments/`date +%Y-%m-%d-%H-%M-%S`-${DATASET}_${NET}_${SESSION}.log
echo "write the experiments detail to file"
cat>${filename}<<EOF
----------basic setting----------
WORKER_NUMBER = $WORKER_NUMBER
DATASET=$DATASET
NET=$NET
PRETRAIN=$PRETRAIN
MAXEPOCHS=$MAXEPOCHS
SESSION=$SESSION
FIXED_LAYER=$FIXED_LAYER
PRETRAIN=$PRETRAIN
WEIGHT_FILE=$WEIGHT_FILE
SAVE_EPOCH=$SAVE_EPOCH

----------learning rate setting----------
OPTIMIZER=$OPTIMIZER
LEARNING_RATE=$LEARNING_RATE
DECAY_STEP=$DECAY_STEP
BATCH_SIZE=$BATCH_SIZE
IS_WARMING_UP=$IS_WARMING_UP
WARMING_UP_LR=$WARMING_UP_LR
BATCH_SIZE=$BATCH_SIZE

----------resume----------
RESUME=$RESUME
RESUME_OPT = $RESUME_OPT  # null for false
RESUME_SESS_EPOCH=$RESUME_SESS_EPOCH #null for false
PRETRAIN=$PRETRAIN
DATASET=$DATASET
NET=$NET
OPTIMIZER=$OPTIMIZER
EOF

LOG=./Experiments/DetailLogs/log-`date +%Y-%m-%d-%H-%M-%S`-${DATASET}-${NET}-${SESSION}.log

# training command
if $IS_WARMING_UP ; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset $DATASET --net $NET \
                   --epochs $MAXEPOCHS --save_epoch=$SAVE_EPOCH \
                   --s $SESSION \
                   --o $OPTIMIZER \
                   --r=$RESUME --resume_opt=$RESUME_OPT --resume_session_epoch=$RESUME_SESS_EPOCH \
                   --fixed_layer=$FIXED_LAYER --pretrain=$PRETRAIN  --weight_file=$WEIGHT_FILE \
                   --checksession $CHECKSESSION --checkepoch $CHECKEPOCH --checkpoint $CHECKPOINT \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --mGPUs --use_tfb --save_model \
                   --wu --wulr $WARMING_UP_LR
                       #2>&1 | tee $LOG $@
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset $DATASET --net $NET \
                   --epochs $MAXEPOCHS --save_epoch=$SAVE_EPOCH \
                   --s $SESSION \
                   --o $OPTIMIZER \
                   --r=$RESUME --resume_opt=$RESUME_OPT --resume_session_epoch=$RESUME_SESS_EPOCH \
                   --fixed_layer=$FIXED_LAYER --pretrain=$PRETRAIN  --weight_file=$WEIGHT_FILE \
                   --checksession $CHECKSESSION --checkepoch $CHECKEPOCH --checkpoint $CHECKPOINT \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --mGPUs --use_tfb --save_model
                       #2>&1 | tee $LOG $@
fi


times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.018  000.018: --- VIM STARTING ---
000.226  000.208: Allocated generic buffers
000.643  000.417: locale set
000.655  000.012: clipboard setup
000.663  000.008: window checked
001.697  001.034: inits 1
001.739  000.042: parsing arguments
001.740  000.001: expanding arguments
001.772  000.032: shell init
002.339  000.567: Termcap init
002.363  000.024: inits 2
002.608  000.245: init highlight
003.614  000.051  000.051: sourcing $HOME/.vimrc
003.648  000.989: sourcing vimrc file(s)
006.199  000.234  000.234: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/getscriptPlugin.vim
006.883  000.466  000.466: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/gzip.vim
007.748  000.606  000.606: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/logiPat.vim
008.141  000.129  000.129: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/manpager.vim
010.211  000.515  000.515: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/matchparen.vim
011.542  001.103  001.103: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/netrwPlugin.vim
012.001  000.143  000.143: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/rrhelper.vim
012.472  000.113  000.113: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/spellfile.vim
013.057  000.351  000.351: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/tarPlugin.vim
013.731  000.274  000.274: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/tohtml.vim
014.525  000.413  000.413: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/vimballPlugin.vim
015.254  000.414  000.414: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/zipPlugin.vim
015.274  006.865: loading plugins
016.009  000.735: loading packages
016.077  000.068: loading after plugins
016.120  000.043: inits 3
017.316  001.196: reading viminfo
036.453  019.137: setup clipboard
036.480  000.027: setting raw mode
036.501  000.021: start termcap
036.526  000.025: clearing screen
037.374  000.848: opening buffers
037.479  000.105: BufEnter autocommands
037.483  000.004: editing files in windows
037.668  000.185: VimEnter autocommands
037.671  000.003: before starting main loop
038.293  000.622: first screen update
038.296  000.003: --- VIM STARTED ---


times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.035  000.035: --- VIM STARTING ---
000.327  000.292: Allocated generic buffers
000.469  000.142: locale set
000.480  000.011: clipboard setup
000.488  000.008: window checked
001.306  000.818: inits 1
001.338  000.032: parsing arguments
001.340  000.002: expanding arguments
001.372  000.032: shell init
001.774  000.402: Termcap init
001.798  000.024: inits 2
002.044  000.246: init highlight
237.408  000.064  000.064: sourcing $HOME/.vimrc
237.495  235.387: sourcing vimrc file(s)
910.802  000.259  000.259: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/getscriptPlugin.vim
1005.791  000.502  000.502: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/gzip.vim
1334.417  000.650  000.650: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/logiPat.vim
1392.620  000.162  000.162: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/manpager.vim
1522.577  000.569  000.569: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/matchparen.vim
1560.944  001.142  001.142: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/netrwPlugin.vim
1605.855  000.166  000.166: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/rrhelper.vim
1675.539  000.181  000.181: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/spellfile.vim
1728.900  000.360  000.360: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/tarPlugin.vim
1804.580  000.367  000.367: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/tohtml.vim
1831.054  000.411  000.411: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/vimballPlugin.vim
1862.972  000.417  000.417: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/zipPlugin.vim
1862.992  1620.311: loading plugins
1950.433  087.441: loading packages
1950.544  000.111: loading after plugins
1950.593  000.049: inits 3
2276.987  327.394: reading viminfo
2297.587  020.600: setup clipboard
2297.616  000.029: setting raw mode
2297.645  000.029: start termcap
2297.674  000.029: clearing screen
2298.511  000.837: opening buffers
2298.601  000.090: BufEnter autocommands
2298.604  000.003: editing files in windows
2298.776  000.172: VimEnter autocommands
2298.778  000.002: before starting main loop
2299.387  000.609: first screen update
2299.390  000.003: --- VIM STARTED ---


times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.033  000.033: --- VIM STARTING ---
000.254  000.221: Allocated generic buffers
123.337  123.083: locale set
123.364  000.027: clipboard setup
123.375  000.011: window checked
269.230  145.855: inits 1
269.291  000.061: parsing arguments
269.294  000.003: expanding arguments
269.329  000.035: shell init
269.770  000.441: Termcap init
269.795  000.025: inits 2
270.042  000.247: init highlight
494.575  000.046  000.046: sourcing $HOME/.vimrc
639.278  369.190: sourcing vimrc file(s)
6894.363  000.339  000.339: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/getscriptPlugin.vim
7178.809  000.481  000.481: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/gzip.vim
7477.896  000.675  000.675: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/logiPat.vim
8019.716  000.176  000.176: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/manpager.vim
8373.647  027.811  027.811: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/matchparen.vim
8623.895  001.236  001.236: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/netrwPlugin.vim
9078.683  000.206  000.206: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/rrhelper.vim
9432.626  000.162  000.162: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/spellfile.vim
9795.830  000.448  000.448: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/tarPlugin.vim
10036.120  000.443  000.443: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/tohtml.vim
10398.939  000.625  000.625: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/vimballPlugin.vim
10750.083  000.556  000.556: sourcing /home/d/denglixi/bin/vim/share/vim/vim81/plugin/zipPlugin.vim
10841.138  10168.702: loading plugins
11055.971  215.833: loading packages
11056.048  000.077: loading after plugins
11056.087  000.039: inits 3
11270.901  214.814: reading viminfo
12742.181  1471.280: setup clipboard
12742.212  000.031: setting raw mode
12742.244  000.032: start termcap
12742.275  000.031: clearing screen
12743.261  000.986: opening buffers
12743.362  000.101: BufEnter autocommands
12743.365  000.003: editing files in windows
12743.546  000.181: VimEnter autocommands
12743.548  000.002: before starting main loop
12744.181  000.633: first screen update
12744.184  000.003: --- VIM STARTED ---
