src/main.o: src/main.cpp src/DeepNet.h src/LogicalCube.h \
  src/sched/DeviceDriver.h src/sched/DeviceHeader.h \
  src/sched/DeviceMemoryPointer.h \
  /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/cblas.h \
  src/sched/../util.h src/sched/../kernels/include.h \
  src/sched/../kernels/lowering.h src/sched/../kernels/mul.h \
  src/sched/../kernels/test.h src/sched/../kernels/conv.h \
  src/sched/../kernels/pool.h src/sched/../kernels/dropout.h \
  src/sched/../kernels/relu.h src/sched/../kernels/lrn.h \
  src/sched/../kernels/softmax.h src/LogicalMatrix.h \
  src/LogicalMatrix_impl.hxx src/LoweringType.h src/LogicalCube_impl.hxx \
  src/Layer.h src/Connector.h src/Report.h src/timer.h \
  src/Connector_impl_Lowering_type1.hxx src/sched/DeviceDriver_CPU.h \
  src/Kernel.h src/Kernel_impl.hxx src/bridges/AbstractBridge.h \
  src/bridges/../Scanner.h src/Scanner_impl.hxx \
  src/bridges/PhysicalOperator.h src/bridges/../parser/cnn.pb.h \
  src/bridges/../algorithms/GradientUpdater.h \
  src/bridges/../algorithms/../parser/corpus.h \
  src/bridges/../parser/parser.h src/bridges/../sched/DeviceDriver_CPU.h \
  src/bridges/MaxPoolingBridge.h src/bridges/MaxPoolingBridge_impl.hxx \
  src/bridges/ReLUBridge.h src/bridges/ReLUBridge_impl.hxx \
  src/bridges/ConvolutionBridge.h src/bridges/ConvolutionBridge_impl.hxx \
  src/bridges/FullyConnectedBridge.h \
  src/bridges/FullyConnectedBridge_impl.hxx \
  src/bridges/SoftmaxLossBridge.h src/bridges/SoftmaxLossBridge_impl.hxx \
  src/bridges/LRNBridge.h src/bridges/LRNBridge_impl.hxx \
  src/bridges/ParallelizedBridge.h src/bridges/PhysicalStratum.h \
  src/bridges/ParallelizedBridge_impl.hxx src/bridges/DropoutBridge.h \
  src/bridges/../DeepNetConfig.h src/bridges/DropoutBridge_impl.hxx \
  src/bridges/FunnelBridge.h src/bridges/FunnelBridge_impl.hxx
