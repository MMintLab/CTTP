# python /home/samanta/CMTJE/joint_embedding_learning/evaluation.py --dataset_name dataset_1 --model_name simclr --run_name dataset_1_run_B_16
# python /home/samanta/CMTJE/joint_embedding_learning/evaluation.py --dataset_name dataset_1 --model_name simclr --run_name dataset_1_run_B_32
# python /home/samanta/CMTJE/joint_embedding_learning/evaluation.py --dataset_name dataset_1 --model_name simclr --run_name dataset_1_run_B_64
# python /home/samanta/CMTJE/joint_embedding_learning/evaluation.py --dataset_name dataset_1 --model_name simclr --run_name dataset_1_run_B_256

# python /home/samanta/CMTJE/joint_embedding_learning/evaluation.py --dataset_name dataset_1 --model_name simclr --run_name dataset_1_run_B_128
# python /home/samanta/CMTJE/joint_embedding_learning/evaluation.py --dataset_name dataset_1 --model_name simclr --run_name pretrained
# python /home/samanta/CMTJE/joint_embedding_learning/evaluation.py --dataset_name dataset_1 --model_name simclr --run_name random_init

python evaluation.py --dataset_name dataset_1 --run_name pretrain1_small_I30200 --device cuda:3 --model_name T3
python evaluation.py --dataset_name dataset_1 --run_name pretrain2_class_small_I30200 --device cuda:3 --model_name T3
python evaluation.py --dataset_name dataset_1 --run_name pretrain2_pose_small_I30200 --device cuda:3 --model_name T3
