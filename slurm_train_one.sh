#!/bin/bash
#SBATCH --partition=jag-urgent --qos=normal
#SBATCH --nodes=1
#SBATCH --job-name=tagparse
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10240
#SBATCH --gres=gpu:1
#SBATCH --output=/sailhome/pengqi/logs/slurm-%j.out
#SBATCH --mail-user=pengqi@cs.stanford.edu
#SBATCH --mail-type=FAIL

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

LANGUAGE=$1
TREEBANK=$2
LC=$3
TB=$4

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PATH=/u/scr/pengqi/anaconda3_slurm/bin:$PATH
export LD_LIBRARY_PATH=/u/scr/pengqi/anaconda3_slurm/lib:$LD_LIBRARY_PATH
cd /u/scr/pengqi/Parser-v3

STACK=stack

mkdir -p $STACK
echo $LANGUAGE $TREEBANK $LC $TB
if [ ! -e $STACK/$LANGUAGE-$TREEBANK ]
then
  touch $STACK/$LANGUAGE-$TREEBANK
  if [ "$LC" == "zh" ]
  then
    pretrained_file=data/embeddings/"$LANGUAGE"T/"$LC".vectors.xz
  elif [ "$LC" == "no" ]
  then
    if [ "$TB" == "bokmaal" ]
    then
      pretrained_file=data/embeddings/"$LANGUAGE"-Bokmaal/"$LC"_bokmaal.vectors.xz
    else  
      pretrained_file=data/embeddings/"$LANGUAGE"-Nynorsk/"$LC"_nynorsk.vectors.xz
    fi
  else
    pretrained_file=data/embeddings/$LANGUAGE/"$LC".vectors.xz
  fi

  basename=CoNLL18/UD_$LANGUAGE-$TREEBANK
  basename1=CoNLL18_peng/UD_$LANGUAGE-$TREEBANK
  train_conllus=data/$basename/"$LC"_$TB-ud-train.conllu
  dev_conllus=data/$basename/"$LC"_$TB-ud-dev.conllu
  if [ ! -e $dev_conllus ]
  then
    dev_conllus=$train_conllus
  fi
  python main.py --save_dir saves/$basename1/Tagger train TaggerNetwork --DEFAULT train_conllus=$train_conllus dev_conllus=$dev_conllus --FormPretrainedVocab pretrained_file=$pretrained_file --force --noscreen
  python main.py --save_dir saves/$basename1/Tagger run TaggerNetwork $train_conllus $dev_conllus

  tagged_train_conllus=saves/$basename1/Tagger/parsed/$train_conllus
  tagged_dev_conllus=saves/$basename1/Tagger/parsed/$dev_conllus
  python scripts/reinsert_compounds.py $train_conllus $tagged_train_conllus
  python scripts/reinsert_compounds.py $dev_conllus $tagged_dev_conllus

  python main.py --save_dir saves/$basename1/Parser train ParserNetwork --DEFAULT train_conllus=$tagged_train_conllus dev_conllus=$tagged_dev_conllus --FormPretrainedVocab pretrained_file=$pretrained_file --force --noscreen
  python main.py --save_dir saves/$basename1/Parser run ParserNetwork $tagged_train_conllu $tagged_dev_conllus

  if [ $? -eq 0 ]
  then
    echo "Success" > $STACK/$LANGUAGE-$TREEBANK
  else
    echo "Failure" > $STACK/$LANGUAGE-$TREEBANK
  fi
fi
