//
//  LogicalCube.h
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include <assert.h>
#include <string>

#ifndef moka_LogicalCube_h
#define moka_LogicalCube_h

typedef float  DataType_SFFloat; /*< Single-precision Floating Point. */
typedef __fp16 DataType_HPFloat; /*< Half-precision Floating Point. */
typedef short  DataType_FPFloat; /*< 16-bit Fixed Point. */

typedef std::string DataType_String; /*< String-type data only for deubgging/unit testing. */

enum LayoutType {
    Layout_CRDB = 0,
    Layout_BDRC = 1
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
 */
template <typename T, LayoutType LAYOUT>
class LogicalCube {
public:

    T * const p_data;
    bool own_data;
    const size_t n_elements;

    const size_t R;
    const size_t C;
    const size_t D;
    const size_t B;

    /**
     * Constructor that points to existing data.
     *  - own_data = False
     **/
    LogicalCube(void * _p_data, size_t _R, size_t _C, size_t _D, size_t _B);

    /**
     * Constuctor that actually allocates the data.
     * If a cube allcoates the data, it needs to
     * free it.
     *  - own_data = True
     **/
    LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B);

    ~LogicalCube();

    /**
     * Get the pointer that points to the physical position
     * corresponding to the logical position (r,c,d,b)
     *
     * This used the trick of paritial specialization as in
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
    void logical_print();
    void physical_print();

    double size_in_GBytes(){
        return 1.0*R*C*D*B*sizeof(T)/1024/1024/1024;
    }

private:

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

};

#include "LogicalCube_impl.hxx"

#endif




