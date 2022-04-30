#!/bin/bash

cd eval
eval_names=$(ls -d suisiann*)
cd ..

result_file="eval/all_results.txt"
echo "==== RESULTS ====" > $result_file

for eval_name in $eval_names
do
	./preprocess-eval-asr.sh $eval_name
	echo "== $eval_name ==" >> $result_file
	echo "dev" >> $result_file
	cat eval/$eval_name/dev_b5.score >> $result_file
	echo "test" >> $result_file
	cat eval/$eval_name/test_b5.score >> $result_file
	echo " " >> $result_file
done


