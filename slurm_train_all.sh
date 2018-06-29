fn=runall_slurm.sh
suffix=one
rm -f $fn
while read -r LANGUAGE TREEBANK LC TB
do
  echo $LANGUAGE $TREEBANK $LC $TB
  echo "sbatch /u/scr/pengqi/Parser-v3/slurm_train_${suffix}.sh $LANGUAGE $TREEBANK $LC $TB 2>/dev/null &" >> $fn
  echo "sleep .1" >> $fn
done < conll18_treebanks.txt

bash /u/scr/pengqi/Parser-v3/$fn
