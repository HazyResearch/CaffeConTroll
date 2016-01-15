//
//  LogicalCube.h
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LogicalCube_h
#define moka_LogicalCube_h

#include "sched/DeviceDriver.h"
#include "sched/DeviceMemoryPointer.h"

#include "LogicalMatrix.h"
#include "LoweringType.h"
#include "util.h"

enum LayoutType {
  Layout_CRDB = 0,
  Layout_BDRC = 1       // SHADJIS TODO: Why don't we use Layout_BDRC ever? Need to benchmark
};

/*
 *       ----          ----
 *    d /   /|   b    /   /|
 *     /   / |  ...  /   / |
 *     ----  /       ----  /
 *   r|    |/       |    |/
 *     ----          ----
 *      c
 *
 * Although not precise, we abuse the name by
 * still calling `r` row, `c` column, and `d`
 * depth. We call `b` batch.
 *
 * 20150303:
 *   Note that, in the context of multiple
 * devices, a LogicalCube should be guarentee'ed
 * to be located in a place that a DeviceDriver
 * can manipulate. This responsibility is currently
 * enforced by the invoker, however, in future,
 * it should be the job of the worker.
 *
 */
template <typename T, LayoutType LAYOUT>
class LogicalCube {
  public:
    const size_t n_elements;

    const size_t R;
    const size_t C;
    const size_t D;
    /*const*/ size_t B; //TODO: need a getter and setter for this
    bool own_data;

    /**
     * Constructor that points to existing data.
     *  - own_data = False
     **/
    LogicalCube(void * _p_data, size_t _R, size_t _C, size_t _D, size_t _B);
    // Device version
    LogicalCube(void * _p_data, size_t _R, size_t _C, size_t _D, size_t _B,
        DeviceDriver * p_driver);

    DeviceMemoryPointer * get_device_pointer(DeviceDriver * p_driver) const {
        return p_driver->get_device_pointer(p_data, n_elements*sizeof(T));
    }

    DeviceMemoryPointer * get_device_pointer_RCDslice(DeviceDriver * p_driver, size_t b_offset,
        size_t num_batches) const {
      #ifdef _DO_ASSERT
        assert(b_offset<B);
        assert(b_offset + num_batches <= B);
      #endif
        return p_driver->get_device_pointer(p_data + b_offset*R*C*D, num_batches*R*C*D*sizeof(T));
    }

    /**
     * Constuctor that actually allocates the data.
     * If a cube allcoates the data, it needs to
     * free it.
     *  - own_data = True
     **/
    LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B);

    /**
     * Constuctor that actually allocates the data
     * ON DEVICE. If a cube allcoates the data,
     * it needs to free it.
     *  - own_data = True
     **/
    LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B, DeviceDriver * p_driver);

    ~LogicalCube();

    T * const get_p_data() const;

    /**
     * Update p_data to point to data.
     * (Note: only allowed if own_data set to false)
     **/
    void set_p_data(T * const data);

    /**
     * Get the pointer that points to the physical position
     * corresponding to the logical position (r,c,d,b)
     *
     * This uses the trick of paritial specialization as in
     * LogicalFetcher. We confirmed that with -O3 opt, both
     * clang and gcc will inline all these functions.
     *
     **/
    T * logical_get(size_t r, size_t c, size_t d, size_t b) const;

    /**
     * Get the pointer that points to the RxC slides of Depth d
     * and Batch b.
     *
     * This operation is only supported for layout CRDB. Any other
     * layout will get a compilation error.
     *
     **/
    T * physical_get_RCDslice(size_t b);

    /**
     * Print to STDOUT the Logical Representation.
     *
     * For example:
     *  BATCH 0 DEPTH 0
     *       a1 d1 g1
     *       b1 e1 h1
     *       c1 f1 i1
     *   BATCH 0 DEPTH 1
     *       a1' d1' g1'
     *       b1' e1' h1'
     *       c1' f1' i1'
     *   BATCH 1 DEPTH 0
     *       a2 d2 g2
     *       b2 e2 h2
     *       c2 f2 i2
     *   BATCH 1 DEPTH 1
     *       a2' d2' g2'
     *       b2' e2' h2'
     *       c2' f2' i2'
     *
     **/
    void logical_print() const;
    void physical_print() const;

    LogicalMatrix<T> get_logical_matrix(size_t depth_index, size_t batch_index) const;

    template<LoweringType LOWERING>
      void remap_output(const size_t O, const size_t B, const size_t kernel_size, DeviceDriver * p_driver);

    void reset_cube();
    void reset_cube(const T val);

    double size_in_GBytes(){
      return 1.0*R*C*D*B*sizeof(T)/1024/1024/1024;
    }

  private:
    T * p_data; // p_data is not const, because we may have to update it per batch
	DeviceMemoryPointer *p_data_device_ptr;
	DeviceDriver *p_data_driver;

    /**
     * Functions used for logical_get for different Layout.
     * For each Layout, we have one such function, that is why TYPECONSTRAINT is void
     **/
    template<LayoutType LAYOUT2, typename TYPECONSTRAINT = void>
      struct LogicalFetcher {};

    /**
     * Functions used forphysical_get_RCslice for diffent Layout.
     * We only support this function for Layout_CRDB; otherwise, throw
     * compilation error.
     **/
    template<LayoutType LAYOUT2, typename TYPECONSTRAINT = typename std::enable_if<LAYOUT2 == Layout_CRDB>::type>
      struct PhysicalFetcher {};

    template<typename TYPECONSTRAINT>
      struct LogicalFetcher<Layout_CRDB, TYPECONSTRAINT> {
        inline static T * logical_get(const LogicalCube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b);
      };

    template<typename TYPECONSTRAINT>
      struct LogicalFetcher<Layout_BDRC, TYPECONSTRAINT> {
        inline static T * logical_get(const LogicalCube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b);
      };

    template<typename TYPECONSTRAINT>
      struct PhysicalFetcher<Layout_CRDB, TYPECONSTRAINT> {
        inline static T * physical_get_RCDslice(const LogicalCube<T, LAYOUT>& cube, size_t b);
      };

    /**
     * LoweringHelper: Use the same trick as before with LogicalFetcher to
     * to handle the logical for lowering the matrix 3 different ways:
     * LOWERING_TYPE1, LOWERING_TYPE2, and LOWERING_TYPE3.
     ****/
    template<LoweringType LOWERING, typename DUMMY = void>
      struct LoweringHelper {};

    template<typename DUMMY>
      struct LoweringHelper<LOWERING_TYPE1, DUMMY> {
        inline static void remap_output(LogicalCube<T, LAYOUT>& cube, const size_t R, const size_t C, const size_t kernel_size, DeviceDriver * p_driver);
      };

    template<typename DUMMY>
      struct LoweringHelper<LOWERING_TYPE2, DUMMY> {

        inline static void remap_output(LogicalCube<T, LAYOUT>& cube, const size_t R, const size_t C, const size_t kernel_size, DeviceDriver * p_driver);
      };

    template<typename DUMMY>
      struct LoweringHelper<LOWERING_TYPE3, DUMMY> {

        inline static void remap_output(LogicalCube<T, LAYOUT>& cube, const size_t R, const size_t C, const size_t kernel_size, DeviceDriver * p_driver);
      };
};

#include "LogicalCube_impl.hxx"

#endif
