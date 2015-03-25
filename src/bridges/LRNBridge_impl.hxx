//
//  LRNBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LRNBridge_impl_hxx
#define moka_LRNBridge_impl_hxx

/**
 * This function is from
 * http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
 **/
inline double fastPrecisePow(double a, double b) {
  int e = (int) b;
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;

  double r = 1.0;
  while (e) {
    if (e & 1) {
      r *= a;
    }
    a *= a;
    e >>= 1;
  }

  return r * u.d;
}

template <typename DataType, typename DriverClass>
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::LRNBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer,
    _p_output_layer, _layer_param, _solver_param, _p_driver),
 alpha(layer_param->lrn_param().alpha()), beta(layer_param->lrn_param().beta()),
 local_size(layer_param->lrn_param().local_size()) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR == iR); assert(oC == iC);
  assert(oB == iB); assert(oD == iD);
  assert(alpha >= 0.);
  assert(beta >= 0.);
  assert(local_size % 2 == 1);
#endif

  denoms = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements LRN in the forward direction. (Note: we only support ACROSS
 * CHANNEL normalization.)
 * This is implemented very differently from Caffe, but it should still
 * produce the same result.
 **/
template <typename DataType, typename DriverClass>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  // Copy input to Device. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal(p_input_layer->p_data_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost = p_driver->get_device_pointer(input_d_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(phost, &plocal);

  report_forward_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);

  _lrn_forward_arg_helper _arg;
  _arg.iR = iR;
  _arg.iC = iC;
  _arg.iD = iD;
  _arg.norm_window = (int) local_size / 2;
  _arg.denoms = (char *) denoms->get_p_data();

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_lrn_forward_arg_helper));

  p_driver->template parallel_map<_f_src_to_dst_lrn_forward,
    _f_lrn_forward>(input, output, sizeof(DataType)*iR*iC*iD, arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // Copy output to Host. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal2(p_output_layer->p_data_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost2 = p_driver->get_device_pointer(output_d_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(&plocal2, phost2);

  // then do normalization (TODO: use parallel_map for this as well)
  DataType * p_input = p_input_layer->p_data_cube->get_p_data();
  DataType * p_output = p_output_layer->p_data_cube->get_p_data();
  DataType * p_denoms = denoms->get_p_data();

  const DataType alpha_over_size = alpha / local_size;
  const size_t n_elements = p_input_layer->p_data_cube->n_elements;
  for (size_t i = 0; i < n_elements; ++i) {
    *p_denoms = alpha_over_size*(*p_denoms) + 1;
#ifdef _FASTPOW
    *p_output = (*p_input)/fastPrecisePow(*p_denoms, beta);
#else
    *p_output = (*p_input)/pow(*p_denoms, beta);
#endif
    p_input++; p_output++; p_denoms++;
  }

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Implements LRN in the backward direction. (Note: we only support ACROSS
 * CHANNEL normalization.)
 * This is also implemented very differently from Caffe, but it should still
 * produce the same result.
 **/
template <typename DataType, typename DriverClass>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {
  report_backward_updateweight_last_transfer.reset();

  p_input_layer->p_gradient_cube->reset_cube();

  const DataType alpha_over_size = alpha / local_size;
  const int norm_window = (int) local_size / 2;

  for (size_t o_b = 0; o_b < iB; ++o_b) {
    for (int o_d = 0; o_d < iD; ++o_d) {
      for (size_t o_c = 0; o_c < iC; ++o_c) {
        for (size_t o_r = 0; o_r < iR; ++o_r) {
          const DataType denom_no_exponent = *denoms->logical_get(o_r, o_c, o_d, o_b);

#ifdef _FASTPOW
          const DataType denom = fastPrecisePow(denom_no_exponent, beta);
#else
          const DataType denom = pow(denom_no_exponent, beta);
#endif
          const DataType denom_n1 = 1.0/(denom*denom_no_exponent);
          const DataType output_grad = *p_output_layer->p_gradient_cube->logical_get(o_r, o_c, o_d, o_b);
          const DataType window_data = *p_input_layer->p_data_cube->logical_get(o_r, o_c, o_d, o_b);

          DataType input_grad;
          DataType input_grad2;
          input_grad2 = beta * denom_n1 * alpha_over_size * 2 * window_data;

          for (int i = -norm_window; i <= norm_window; ++i) {
            const int channel = o_d + i;
            if (channel < 0 || channel >= iD) {
              continue; // in the padding region, so we're adding 0
            }
            const DataType input_data = *p_input_layer->p_data_cube->logical_get(o_r, o_c, channel, o_b);

            if (i==0) {
              input_grad = 1.0/denom -  input_grad2 * input_data;
            } else {
              input_grad = - input_grad2 * input_data;
            }

            *p_input_layer->p_gradient_cube->logical_get(o_r, o_c, channel, o_b) += input_grad * output_grad;
          }
        }
      }
    }
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~LRNBridge() {
  delete denoms;
}

#endif
