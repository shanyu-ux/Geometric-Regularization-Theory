/**
 * NCGD: Non-Commutative Geometric Dynamics
 * Core Module: SE(3) Lie Algebra Implementation
 * * This module implements exact exponential maps and Lie brackets for SE(3)
 * to ensure geometric rigidity in optimization, contrasting with 
 * first-order Euclidean approximations.
 * * Mathematical Ref: Chirikjian, G. S. "Stochastic Models, Information Theory, and Lie Groups".
 */

#ifndef NCGD_SE3_ALGEBRA_HPP
#define NCGD_SE3_ALGEBRA_HPP

#include <array>
#include <cmath>
#include <stdexcept>

namespace ncgd {

    // R^6 Tangent Vector: [omega (3), v (3)]
    using TangentVector = std::array<double, 6>;
    using Matrix3x3 = std::array<std::array<double, 3>, 3>;
    using Matrix4x4 = std::array<std::array<double, 4>, 4>;

    class SE3Algebra {
    public:
        /**
         * Computes the Lie Bracket [xi1, xi2] = ad_{xi1}(xi2).
         * * The Lie bracket quantifies the non-commutativity of the group.
         * For SE(3), this captures the coupling between rotation and translation
         * (e.g., Coriolis effects) absent in vector addition.
         * * Formula: [(w1, v1), (w2, v2)] = (w1 x w2, w1 x v2 - w2 x v1)
         */
        static TangentVector lie_bracket(const TangentVector& xi1, const TangentVector& xi2) {
            // Extract angular (w) and linear (v) components
            auto w1 = get_angular(xi1);
            auto v1 = get_linear(xi1);
            auto w2 = get_angular(xi2);
            auto v2 = get_linear(xi2);

            // Compute cross products
            auto w_new = cross_product(w1, w2);
            auto term1 = cross_product(w1, v2);
            auto term2 = cross_product(w2, v1);

            // Linear component: w1 x v2 - w2 x v1
            std::array<double, 3> v_new = {
                term1[0] - term2[0],
                term1[1] - term2[1],
                term1[2] - term2[2]
            };

            return {w_new[0], w_new[1], w_new[2], v_new[0], v_new[1], v_new[2]};
        }

    private:
        static std::array<double, 3> get_angular(const TangentVector& tv) {
            return {tv[0], tv[1], tv[2]};
        }
        
        static std::array<double, 3> get_linear(const TangentVector& tv) {
            return {tv[3], tv[4], tv[5]};
        }

        static std::array<double, 3> cross_product(const std::array<double, 3>& a, 
                                                   const std::array<double, 3>& b) {
            return {
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            };
        }
    };
}

#endif // NCGD_SE3_ALGEBRA_HPP
