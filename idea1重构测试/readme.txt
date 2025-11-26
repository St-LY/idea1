#训练
python train_vfl_simple.py --dataset MNIST --epochs 5
python train_vfl_simple.py --dataset FashionMNIST --epochs 5
python train_vfl_simple.py --dataset CIFAR10 --epochs 10


#黑盒攻击
python attack_blackbox.py --vectors_file shuffle_defense_vectors/vectors/epoch_5.pkl --dataset MNIST --target_client 0
python attack_blackbox.py --vectors_file shuffle_defense_vectors/vectors/epoch_5.pkl --dataset FashionMNIST --target_client 0
python attack_blackbox.py --vectors_file shuffle_defense_vectors/vectors/epoch_5.pkl --dataset CIFAR10 --target_client 0


#白盒攻击
python attack_whitebox.py --vectors_file shuffle_defense_vectors/vectors/epoch_10.pkl --model_dir shuffle_defense_vectors/models --dataset CIFAR10 --target_client 0
