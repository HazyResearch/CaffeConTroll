SHADJIS TODO: This is a very old list, need to update with new prototxt format

The following features are unsupported:


```
message FillerParameter {
  optional int32 sparse = 7 [default = -1];
}
message SolverParameter {
  optional bool test_initialization = 32 [default = true];
  optional int32 display = 6;
  Display the cost averaged over the last average_cost iterations
  optional int32 average_loss = 33 [default = 1];
  optional string lr_policy = 8;
  optional float gamma = 9;
  optional float power = 10;
  repeated int32 stepvalue = 34;
   optional bool snapshot_diff = 16 [default = false];
   the mode solver will use: 0 for CPU and 1 for GPU. Use GPU in default.
   enum SolverMode {
     CPU = 0;
     GPU = 1;
   }
   optional SolverMode solver_mode = 17 [default = GPU];
   optional int32 device_id = 18 [default = 0];
   optional int64 random_seed = 20 [default = -1];
  enum SolverType {
    NESTEROV = 1;
    ADAGRAD = 2;
  }
  optional float delta = 31 [default = 1e-8];
}
message LayerParameter {
  enum LayerType {
    NONE = 0;
    ABSVAL = 35;
    ACCURACY = 1;
    ARGMAX = 30;
    BNLL = 2;
    CONCAT = 3;
    CONTRASTIVE_LOSS = 37;
    DUMMY_DATA = 32;
    EUCLIDEAN_LOSS = 7;
    ELTWISE = 25;
    FLATTEN = 8;
    HDF5_DATA = 9;
    HDF5_OUTPUT = 10;
    HINGE_LOSS = 28;
    IM2COL = 11;
    IMAGE_DATA = 12;
    INFOGAIN_LOSS = 13;
    MEMORY_DATA = 29;
    MULTINOMIAL_LOGISTIC_LOSS = 16;
    MVN = 34;
    POWER = 26;
    SIGMOID = 19;
    SIGMOID_CROSS_ENTROPY_LOSS = 27;
    SILENCE = 36;
    SOFTMAX = 20;
    SPLIT = 22;
    SLICE = 33;
    TANH = 23;
    WINDOW_DATA = 24;
    THRESHOLD = 31;
  }
  optional AccuracyParameter accuracy_param = 27;
  optional ArgMaxParameter argmax_param = 23;
  optional ConcatParameter concat_param = 9;
  optional ContrastiveLossParameter contrastive_loss_param = 40;
  optional DummyDataParameter dummy_data_param = 26;
  optional EltwiseParameter eltwise_param = 24;
  optional HDF5DataParameter hdf5_data_param = 13;
  optional HDF5OutputParameter hdf5_output_param = 14;
  optional HingeLossParameter hinge_loss_param = 29;
  optional ImageDataParameter image_data_param = 15;
  optional InfogainLossParameter infogain_loss_param = 16;
  optional MemoryDataParameter memory_data_param = 22;
  optional MVNParameter mvn_param = 34;
  optional PowerParameter power_param = 21;
  optional SigmoidParameter sigmoid_param = 38;
  optional SliceParameter slice_param = 31;
  optional TanHParameter tanh_param = 37;
  optional ThresholdParameter threshold_param = 25;
  optional WindowDataParameter window_data_param = 20;
}
message ConvolutionParameter {
  optional uint32 pad_h = 9 [default = 0];
  optional uint32 pad_w = 10 [default = 0];
  optional uint32 kernel_h = 11;
  optional uint32 kernel_w = 12;
  optional uint32 stride_h = 13;
  optional uint32 stride_w = 14;
}
message DataParameter {
  enum DB {
    LEVELDB = 0;
  }
  optional DB backend = 8 [default = LEVELDB];
}
message PoolingParameter {
  enum PoolMethod {
    AVE = 1;
    STOCHASTIC = 2;
  }
  optional uint32 pad_h = 9 [default = 0];
  optional uint32 pad_w = 10 [default = 0];
  optional uint32 kernel_h = 5;
  optional uint32 kernel_w = 6;
  optional uint32 stride_h = 7;
  optional uint32 stride_w = 8;
}
message ReLUParameter {
  optional float negative_slope = 1 [default = 0];
}
```
