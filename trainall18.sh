# !/bin/bash
# This goes in ..

export CUDA_VISIBLE_DEVICES=$1
source tf3/bin/activate
cd Parser-v3

STACK=stack

mkdir -p $STACK
#mkdir -p /scr/tdozat/v3saves
#ln -s /scr/tdozat/v3saves saves
while read -r LANGUAGE TREEBANK LC TB
do
  echo $LANGUAGE $TREEBANK $LC $TB
  if [ ! -e $STACK/$LANGUAGE-$TREEBANK ] 
  then
    touch $STACK/$LANGUAGE-$TREEBANK
    if [ "$LC" == "zh" ]
    then
      pretrained_file=data/embeddings/"$LANGUAGE"T/"$LC".vectors.xz
    else
      pretrained_file=data/embeddings/$LANGUAGE/"$LC".vectors.xz
    fi

    basename=CoNLL18/UD_$LANGUAGE-$TREEBANK
    train_conllus=data/$basename/"$LC"_$TB-ud-train.conllu
    dev_conllus=data/$basename/"$LC"_$TB-ud-dev.conllu
    if [ ! -e $dev_conllus ]
    then
      dev_conllus=$train_conllus
    fi
    python main.py --save_dir saves/$basename/Tagger train TaggerNetwork --DEFAULT train_conllus=$train_conllus dev_conllus=$dev_conllus --FormPretrainedVocab pretrained_file=$pretrained_file
    python main.py --save_dir saves/$basename/Tagger run TaggerNetwork $train_conllus $dev_conllus

    tagged_train_conllus=saves/$basename/Tagger/parsed/$train_conllus
    tagged_dev_conllus=saves/$basename/Tagger/parsed/$dev_conllus
    python scripts/reinsert_compounds.py $train_conllus $tagged_train_conllus
    python scripts/reinsert_compounds.py $dev_conllus $tagged_dev_conllus

    python main.py --save_dir saves/$basename/Parser train ParserNetwork --DEFAULT train_conllus=$tagged_train_conllus dev_conllus=$tagged_dev_conllus --FormPretrainedVocab pretrained_file=$pretrained_file
    python main.py --save_dir saves/$basename/Parser run ParserNetwork $tagged_train_conllu $tagged_dev_conllus

    if [ $? -eq 0 ] 
    then 
      echo "Success" > $STACK/$LANGUAGE-$TREEBANK
    else 
      echo "Failure" > $STACK/$LANGUAGE-$TREEBANK
    fi 
  fi
done < ../conll18_treebanks.txt

cd ..
