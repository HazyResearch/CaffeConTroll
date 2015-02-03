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
  // only kernel size and num_output_features don't have default values
  BridgeConfig(const size_t _k_size, const size_t _num_output_features,
      const size_t _padding = 0, const size_t _stride = 1,
      const bool _bias_term = true, const InitializerType _weight_initializer = CONSTANT,
      const InitializerType _bias_initializer = CONSTANT) : kernel_size(_k_size),
      num_output_features(_num_output_features), padding(_padding), stride(_stride),
      bias_term(_bias_term), weight_initializer(_weight_initializer),
      bias_initializer(_bias_initializer) {}

  size_t kernel_size;
  size_t num_output_features;
  size_t padding;
  size_t stride;
  bool   bias_term;
  InitializerType weight_initializer;
  InitializerType bias_initializer;
};

#endif
