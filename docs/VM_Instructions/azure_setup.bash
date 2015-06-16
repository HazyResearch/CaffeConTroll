# Run with ./azure_setup.bash
# Tested on Azure Ubuntu D-Series

sudo apt-get install make g++ unzip gfortran libprotobuf-dev libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
mkdir data
cd data
mkdir ilsvrc12_train_1000_lmdb
cd ilsvrc12_train_1000_lmdb
wget i.stanford.edu/hazy/share/data.mdb
cd ../../
wget https://github.com/HazyResearch/CaffeConTroll/archive/master.zip
unzip *zip
cd CaffeConTroll-master
mkdir externals
cd externals
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.14.zip
unzip *zip
cd OpenBLAS-0.2.14/
make clean && make -j
make PREFIX=~/CaffeConTroll-master/externals/OpenBLAS-0.2.14/
export LD_LIBRARY_PATH=~/CaffeConTroll-master/externals/OpenBLAS-0.2.14/:$LD_LIBRARY_PATH
cd ../../
cp config.sample.ubuntu_vm .config
make clean && make -j all
