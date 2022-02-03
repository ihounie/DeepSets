# bash script for running code

# check if GPU id passed in argument
GPU_ID=$1
re='^[0-3]+$'
if ! [[ $1 =~ $re ]] ; then
   GPU_ID=0
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo 'Using GPU: '${GPU_ID}


# LDA path
ROOT='./data/'
#INPUT_DATA=${ROOT}'lda/lda_top_words_split_allwords_1k.json'
INPUT_DATA=${ROOT}'lda/lda_top_words_split_allwords_3k.json'
#INPUT_DATA=${ROOT}'lda/lda_top_words_split_allwords_5k.json'

# lda
MODEL='w2v_sum'
#MODEL='w2v_GCN'
#LR = 0.00001
LR=0.00001
MARGIN=0.1
# training
python -u train.py -inputData ${INPUT_DATA} -margin ${MARGIN} -modelName ${MODEL} -dataset lda -learningRate $LR -evalSize 5 -evalPerEpoch 50 -numEpochs 100 -embedSize 50 -dropout 1
