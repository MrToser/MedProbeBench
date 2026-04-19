nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/MedProbeBench_CentralNervousSystem_v0.4.jsonl >1.txt 2>&1 &
nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/MedProbeBench_Digestive_v0.4.jsonl >2.txt 2>&1 &
nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/MedProbeBench_Hematolymphoid_v0.4.jsonl >3.txt 2>&1 &
nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/MedProbeBench_SoftTissueBone_v0.4.jsonl >4.txt 2>&1 &
nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/MedProbeBench_ThoracicCancer_v0.4.jsonl >5.txt 2>&1 &


# nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/MedProbeBench_Hematolymphoid_v0.4_21.jsonl >1.txt 2>&1 &
# nohup bash run_evaluation_1.sh -a -n 1 -d ./datasets/MedProbeBench_Hematolymphoid_v0.4_2.jsonl >2.txt 2>&1 &
# nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/test_judge_model_1.jsonl >1.txt 2>&1 &
# nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/test_judge_model_2.jsonl >2.txt 2>&1 &
# nohup bash run_evaluation_1.sh -a -n 12 -d ./datasets/test_judge_model_3.jsonl >3.txt 2>&1 &


bash run_evaluation.sh -a -n 1 -d ./datasets/MedProbeBench.jsonl -s 0305