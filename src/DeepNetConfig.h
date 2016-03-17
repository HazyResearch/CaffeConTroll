
#ifndef Deepnet_Config_hxx
#define Deepnet_Config_hxx

class DeepNetConfig {
  friend class DeepNet; // so that DeepNet can access train_

  protected:
    static bool train_;

  public:
    static bool train() { return train_; }
    static void setTrain(bool t) { train_ = t; }
};

#endif
