//
//  BridgeConfig.h
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Bridge_Config_h
#define moka_Bridge_Config_h

struct BridgeConfig {
  // default values for kernel size, stride, and padding
  BridgeConfig() : kernel_size(1), stride(1), padding(0) {}
  BridgeConfig(const size_t _k_size) : kernel_size(_k_size), stride(1), padding(0) {}
  BridgeConfig(const size_t _k_size, const size_t _stride)
    : kernel_size(_k_size), stride(_stride), padding(0) {}
  BridgeConfig(const size_t _k_size, const size_t _stride, const size_t _padding)
    : kernel_size(_k_size), stride(_stride), padding(_padding) {}

  size_t kernel_size;
  size_t stride;
  size_t padding;
};

#endif
