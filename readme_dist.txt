Running the distributed version

1. Get the CIFAR-10 dataset by running:
  ./get_datasets.sh
2 Compilte the distributed version:
   make dist
3. Run the following network:
Scheduler:
DataBridge(Python)-->ConvBridge(C++)-->ConvBridge(C++)-->FCBridge(Python)-->FCBridge(Python)
-->LossBridge(Python)

	mpirun -np 2 python pydist.py : -np 1 ./dist 200 3 32 32 6 3 2 1 : -np 1 ./dist 200 6 16 16 24 3 2 1 : -np 3 python pydist.py