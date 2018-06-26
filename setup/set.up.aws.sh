#!/bin/bash

read -p "(1) create p2.xlarge AWS instance with Deep Learning AMI (Ubuntu)"

read -p "(2) scp ~/.ssh/ml_aws_github_rsa to the instance: scp -i ~/.aws/ml_aws_key.pem ~/.ssh/ml_aws_github_rsa ubuntu@<ipaddress>:.ssh/."

read -p "(3) scp notMNIST.pickle to the instance: scp -i ~/.aws/ml_aws_key.pem ../udacity-730/data/notMNIST.pickle ubuntu@<ipaddress>:."

read -p "(4) ssh to the instance:  ssh -i ~/.aws/ml_aws_key.pem ubuntu@<ipaddress>"

read -p "(5) run the below script ..."


cat >~/.ssh/config <<EOL
Host *
 UseKeychain yes
 AddKeysToAgent yes

IdentityFile ~/.ssh/ml_aws_github_rsa
EOL

# ssh-add ~/.ssh/ml_aws_github_rsa
git clone git@github.com:rolfrussell/ml_playground.git
cp ml_playground/setup/.bash_rolf ~/.
cp ml_playground/setup/.tmux.conf ~/.
mv notMNIST.pickle ml_playground/udacity-730/data/.

echo ". $HOME/.bash_rolf" >> ~/.bashrc
exec bash  # reload .bashrc

conda activate tensorflow_p36
