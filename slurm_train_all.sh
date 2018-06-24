rm -f runall_slurm.sh
while read -r LANGUAGE TREEBANK LC TB
do
  echo $LANGUAGE $TREEBANK $LC $TB
  echo "sbatch /u/scr/pengqi/Parser-v3/slurm_train_one.sh $LANGUAGE $TREEBANK $LC $TB 2>/dev/null &" >> runall_slurm.sh
  echo "sleep .1" >> runall_slurm.sh
done < conll18_treebanks.txt

#bash /u/scr/pengqi/Parser-v3/runall_slurm.sh
