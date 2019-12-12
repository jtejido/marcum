/*
 * Copyright (C) 2019 Jericko Tejido
 * ------------------------------
 * This was ported from LALSuite:
 * Copyright (C) 2014 Kipp Cannon
 *
 * Implementation of the algorithm described in
 *
 * A. Gil, J. Segura, and N. M. Temme.  Algorithm 939:  Computation of the
 * Marcum Q-Function.  ACM Transactions on Mathematical Software (TOMS),
 * Volume 40 Issue 3, April 2014, Article No. 20. arXiv:1311.0681
 *
 * with a few modifications.  In particular, a different expression is used
 * here for the 0th term in the asymptotic expansion for large (xy) than
 * shown in Section 4.1 such that it remains numerically stable in the x ~=
 * y limit, and a special case of the recursion implemented that remains
 * valid when x = y.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package marcum

import (
	"fmt"
	gsl "github.com/jtejido/ggsl"
	integ "github.com/jtejido/ggsl/integration"
	sf "github.com/jtejido/ggsl/specfunc"
	"math"
)

/*
 * ============================================================================
 *
 *                                 Support Code
 *
 * ============================================================================
 */

/*
 * equation (8), normalized incomplete Gamma function,
 *
 *	Q_{\mu}(x) = \Gamma(\mu, x) / \Gamma(\mu).
 *
 * we use the GSL implementations, but Gil et al.'s tests of their Marcum Q
 * function implemention rely on their own implementation of these
 * functions.  GSL's implementations might not work as well as theirs and
 * so the Marcum Q implemented here might not meet their accuracy claims.
 * on the other hand, GSL might be using exactly their aglorithm, or it
 * might be using an even better one, who knows.
 *
 * their algorithm is described in
 *
 * Gil , A., Segura , J., and Temme , N. M. 2012. Efficient and accurate
 * algorithms for the computation and inversion of the incomplete gamma
 * function ratios. SIAM J. Sci. Comput.  34(6), A2965–A2981.
 * arXiv:1306.1754.
 */

func q(mu, x float64) float64 {
	return sf.Gamma_inc_Q(mu, x)
}

/*
 * \sqrt{x / y} * equation (14).  see
 *
 * W. Gautschi, J. Slavik, On the Computation of Modified Bessel Function
 * Ratios, Mathematics of Computation, Vol. 32, No. 143 (Jul., 1978), pp.
 * 865-875.
 *
 * for information on using continued fractions to compute the ratio
 * directly.
 */

func cMu(mu, xi float64) float64 {
	return sf.Bessel_Inu_scaled(mu, xi) / sf.Bessel_Inu_scaled(mu-1., xi)
}

/*
 * log of equation (32).  computed using log \Gamma functions from GSL.
 */

func lnA(n int, mu float64) float64 {
	mu += 0.5
	return sf.Lngamma(mu+float64(n)) - sf.Lngamma(mu-float64(n)) - float64(n)*math.Log(2.) - sf.Lngamma(float64(n)+1)
}

/*
 * equation (84).  \zeta^{2} / 2 = phi(xi) - phi(z0), with special case for
 * y ~= x + 1 given by expansion in (85) and (86).
 */

func halfZeta2(x, y float64) float64 {
	if math.Abs(y-x-1.) < 1e-3 {
		c := []float64{
			1.,
			-(3.*x + 1.) / 3.,
			((72.*x+42.)*x + 7.) / 36.,
			-(((2700.*x+2142.)*x+657.)*x + 73.) / 540.,
			((((181440.*x+177552.)*x+76356.)*x+15972.)*x + 1331.) / 12960.,
		}
		z := (y - x - 1.) / math.Pow(2.*x+1., 2.)
		z_to_k_p_1 := z
		sum := z_to_k_p_1

		for k := 1; k < 5; k++ {
			z_to_k_p_1 *= z
			sum += c[k] * z_to_k_p_1
		}
		return math.Pow(2.*x+1., 3.) * sum * sum / 2.
	} else {
		root_1_plus_4xy := math.Sqrt(1. + 4.*x*y)
		return x + y - root_1_plus_4xy + math.Log((1.+root_1_plus_4xy)/(2.*y))
	}
}

/*
 * equation (56).  = \sqrt{2 (phi(xi) - phi(z0))} with the sign of x + 1 -
 * y.  phi(xi) - phi(z0) is given by the expression in equation (84).
 */

func zeta(x, y float64) float64 {
	return math.Copysign(math.Sqrt(2.*halfZeta2(x, y)), x+1.-y)
}

/*
 * equation (98).  note:  rho, r, and r'sin(theta) are all even functions
 * of theta.  we assume 0 <= theta <= pi
 */

func thetaOverSinTheta(theta float64) float64 {
	/* Taylor series = 1 + theta^2 / 6 + 7 theta^4 / 360.  therefore,
	 * once theta is below 10^-4, the first two terms are sufficient */
	if theta < 1e-4 {
		return theta*theta/6. + 1.
	}

	return theta / math.Sin(theta)
}

func thetaOverSinThetaPrimedSinTheta(theta float64) float64 {
	/* sin x * d/dx (x / sin x)
	 *	= (sin x - x cos x) / sin x,
	 *	= 1 - x / tan x
	 *
	 * Taylor series is
	 *
	 * 	theta^2 / 3 + theta^4 / 45 + 2 theta^6 / 945 + ...
	 *
	 * therefore once theta is below 10^-4 the first two terms are
	 * sufficient
	 */
	if theta < 1e-4 {
		return (theta*theta/45. + 1./3.) * theta * theta
	}

	return 1. - theta/math.Tan(theta)
}

func rho(theta_over_sin_theta_, xi float64) float64 {
	return math.Sqrt(theta_over_sin_theta_*theta_over_sin_theta_ + xi*xi)
}

func r(theta, y, xi float64) float64 {
	theta_over_sin_theta_ := thetaOverSinTheta(theta)

	xi /= theta_over_sin_theta_
	return (1. + math.Sqrt(1.+xi*xi)) * theta_over_sin_theta_ / (2. * y)
}

func rPrimedSinTheta(theta, y, xi float64) float64 {
	theta_over_sin_theta_ := thetaOverSinTheta(theta)

	xi /= theta_over_sin_theta_
	return (1. + 1./math.Sqrt(1.+xi*xi)) * thetaOverSinThetaPrimedSinTheta(theta) / (2. * y)
}

/*
 * equation (96).  f is an even function of theta.
 *
 * NOTE:  there is a typo in both the final printed version and in the copy
 * on arXiv, the middle term in the denominator should read "- 2 r(\theta)
 * \cos(\theta)".
 *
 * we rewrite the denominator as (r - cos(theta))^2 + 1 - cos^2(theta) =
 * ((r - cos(theta))^2 + sin^2(theta)
 */
func f(theta, y, xi float64) float64 {
	r_ := r(theta, y, xi)
	r_minus_cos_theta := r_ - math.Cos(theta)
	sin_theta := math.Sin(theta)

	return (rPrimedSinTheta(theta, y, xi) - r_minus_cos_theta*r_) / (r_minus_cos_theta*r_minus_cos_theta + sin_theta*sin_theta)
}

/*
 * equation (97).  psi is an even function of theta.
 */
func psi(theta, xi float64) float64 {
	theta_over_sin_theta_ := thetaOverSinTheta(theta)
	rho_ := rho(theta_over_sin_theta_, xi)
	root_xi2_plus_1 := math.Sqrt(1. + xi*xi)

	return math.Cos(theta)*rho_ - root_xi2_plus_1 - math.Log((theta_over_sin_theta_+rho_)/(1.+root_xi2_plus_1))
}

/*
 * equation (100), unscaled variables
 */

func f1f2(x, M float64) (f1 float64, f2 float64) {
	a := x + M
	b := math.Sqrt(4.*x + 2.*M)
	f1 = a - b
	f2 = a + b
	return
}

/*
 * ============================================================================
 *
 *                         Implementations by Domain
 *
 * ============================================================================
 */

/*
 * series expansion described in section 3.  equation (7).
 */

func marcumQSmallX(M, x, y float64) float64 {
	n := 0.
	x_to_n_over_n_factorial := 1.
	sum := 0.

	for {
		term := x_to_n_over_n_factorial * q(M+n, y)
		sum += term
		if term <= 1e-17*sum {
			break
		}
		n++
		x_to_n_over_n_factorial *= x / n
	}

	return math.Exp(-x) * sum
}

/*
 * asymptotic expansions in section 4.1.  equation (37) for y > x, 1 -
 * equation (39) for x < y, with modifications for x == y.
 */

func marcumQLargeXY(M, x, y, xi float64) float64 {
	var Psi, sum, Phi float64

	root_y_minus_root_x := math.Sqrt(y) - math.Sqrt(x)
	/* equation (31) */
	sigma := root_y_minus_root_x * root_y_minus_root_x / xi
	rho_ := math.Sqrt(y / x)

	rho_to_M_over_root_8_pi := math.Pow(y/x, M/2.) / math.Sqrt(8.*math.Pi)
	e_to_neg_sigma_xi_over_xi_to_n_minus_half := math.Exp(-root_y_minus_root_x*root_y_minus_root_x) * math.Sqrt(xi)

	/* Phi_{0}.  equation (35).  NOTE:  that in (35) they show
	 * \sqrt{\sigma \xi} being replaced by \sqrt{y} - \sqrt{x}, but it
	 * should be |\sqrt{y} - \sqrt{x}|.  the x ~= y limit is obtained
	 * from
	 *
	 *	erfc(a x) / x = 1/x - 2/\sqrt{pi} [ a / (1 0!) - a^3 x^2 / (3 1!) + a^5 x^4 / (5 2!) - a^7 x^6 / (7 3!) + ...]
	 *
	 * the ratio |2nd term| / |0th term| is < 10^-17 when ax < 3 *
	 * 10^-6.  using this,
	 *
	 *	Phi_{0} = sqrt(pi / sigma) - 2 sqrt(xi)
	 *
	 * to the limits of double precision when |sqrt(y)-sqrt(x)| < 3e-6.
	 * we cheat a bit and call that 1e-5 because our final answer isn't
	 * going to be perfect anyway;  and we compute the ratio of
	 * sqrt(pi)/sqrt(sigma), instead of sqrt(pi/sigma), to skirt
	 * numerical overflow until sigma is identically 0.
	 *
	 * the special case for \sigma = 0 is found by substituting
	 * \Phi_{0} from (35) into (36) and determining the desired value
	 * of \Phi_{1}.  when this is done we find a net positive power of
	 * \sigma in the term involving \Phi_{0} and so evidently the
	 * correct value for \Phi_{1} will be obtained by setting \Phi_{0}
	 * to any finite value and allowing the factor of \sigma appearing
	 * in (36) to eliminate it naturally when \sigma = 0 (if we let
	 * \Phi_{0} diverge we end up with inf*0 = nan and the code fails).
	 */
	if math.Abs(root_y_minus_root_x) < 1e-5 {
		if sigma != 0. {
			Phi = math.Sqrt(math.Pi)/math.Sqrt(sigma) - 2.*math.Sqrt(xi)
		}
	} else {
		Phi = math.Sqrt(math.Pi/sigma) * sf.Erfc(math.Abs(root_y_minus_root_x))
	}

	/* Psi_{0} from Phi_{0}.  this is a special case of equation (38)
	 * that remains valid in the small \rho-1 limit (x ~= y).  when n =
	 * 0, from equation (32) we see that A_{0}(\mu) = 1, and
	 * equation (38) can be written
	 *
	 *	\Psi_{0} = \rho^{\mu - 1} / \sqrt{8\pi} (\rho-1) * \Phi_{0}.
	 *
	 * meanwhile, from equation (31) we have that
	 *
	 *	\sqrt{\sigma} = |\rho - 1| / \sqrt{2 \rho}
	 *
	 * and so equation (35) can be written
	 *
	 *	 \Phi_{0} = \sqrt{2 \pi \rho} / |\rho - 1| erfc(\sqrt{y} - \sqrt{x}).
	 *
	 * combining these we find
	 *
	 *	\Psi_{0} = \rho^{\mu - .5} / 2 erfc(\sqrt{y} - \sqrt{x}) * sgn(\rho - 1).
	 *
	 * NOTE:  this result is only used for x < y (\rho > 1).  for x > y
	 * (\rho < 1) we compute Q=1-P where from equation (39) -P is the
	 * sum of -\tilde{\Psi}_{n} = \Psi_{n} except -\tilde{\Psi}_{0}
	 * which is given by -1 * equation (40).  comparing -1 * (40) to
	 * our result above when \rho < 1, we see that they are nearly
	 * identical except for the sign of the argument of erfc() being
	 * flipped (making it positive, again).  therefore, by taking the
	 * absolute value of the erfc() argument we obtain an expression
	 * for the 0th term in the sum valid in both cases.  the only
	 * remaining difference is that the sum needs to be initialized to
	 * +1 in the x > y case and 0 in the x < y case.
	 *
	 * if rho is exactly equal to 1, instead of relying on FPUs
	 * agreeing that 1-1=+0 and not -0 we enter the result with the
	 * correct sign by hand.  the "correct" sign is decided by the
	 * initialization of the sum.  we've chosen to initialize the x==y
	 * case to 0, so the correct sign for Psi_{0} is positive.
	 */
	if rho_ == 1. {
		Psi = .5
	} else {
		Psi = math.Copysign(math.Pow(rho_, M-.5)*sf.Erfc(math.Abs(root_y_minus_root_x))/2., rho_-1.)
	}

	if x > y {
		sum = 1.
	}

	n := 0

	for {
		/* equation (37) */
		sum += Psi
		if math.Abs(Psi) <= 1e-17*math.Abs(sum) {
			break
		}

		/* next n.  rho_to_M_over_root_8_pi is providing the factor
		 * of (-1)^n, so we flip its sign here, too.  */
		n++
		rho_to_M_over_root_8_pi = -rho_to_M_over_root_8_pi
		e_to_neg_sigma_xi_over_xi_to_n_minus_half /= xi

		/* Phi_{n} from Phi_{n - 1}.  equation (36) */
		Phi = (e_to_neg_sigma_xi_over_xi_to_n_minus_half - sigma*Phi) / (float64(n) - .5)

		/* Psi_{n} from Phi_{n}.  equation (38) */
		lnA_n_M_minus_1 := lnA(n, M-1.)
		Psi = rho_to_M_over_root_8_pi * math.Exp(lnA_n_M_minus_1) * (1. - math.Exp(lnA(n, M)-lnA_n_M_minus_1)/rho_) * Phi
	}

	/*
	 * for very small probabilities, improper cancellation can leave us
	 * with a small negative number.  nudge back into the allowed range
	 */
	if -1e-200 < sum && sum < 0. {
		sum = 0.
	}

	return sum
}

/*
 * recurrence relation in equation (14).
 */
func marcumQRecurrence(M, x, y, xi float64) float64 {
	/* factor in equation (14) we've omitted from our c_mu() */
	root_y_over_x := math.Sqrt(y / x)

	/* where to start evaluating recurrence relation */
	mu := math.Sqrt(2.*xi) - 1.
	/* make sure it differs from M by an integer */
	mu = M - math.Ceil(M-mu)

	Qmu_minus_1 := MarcumQModified(mu-1., x, y)
	Qmu := MarcumQModified(mu, x, y)

	for ok := true; ok; ok = mu < M {
		cmu := root_y_over_x * cMu(mu, xi)

		/* compute Q_{\mu + 1} from Q_{\mu} and Q_{\mu - 1} */
		Qmu_saved := Qmu
		Qmu = (1.+cmu)*Qmu - cmu*Qmu_minus_1
		Qmu_minus_1 = Qmu_saved
		mu++
	}

	if math.Abs(mu-M)/M > 1e-15 {
		panic(fmt.Sprintf("recursion failed to land on correct order:  wanted M=%v, got M=%v", M, mu))
	}

	return Qmu
}

/*
 * asymptotic expansion in section 4.2
 */
func marcumQLargeM(M, x, y float64) float64 {
	/* equation (56) */
	zeta_ := zeta(x, y)

	/* equation (67).  if size of Psi array is changed, updated safety
	 * check inside loop below */
	e_to_neg_half_M_zeta_2 := math.Exp(-M * halfZeta2(x, y))
	var Psi [20]float64
	Psi[0] = math.Sqrt(math.Pi/(2.*M)) * sf.Erfc(-zeta_*math.Sqrt(M/2.))
	Psi[1] = e_to_neg_half_M_zeta_2 / M

	sum := 0.
	k := 1

	for {
		/* equation (71) */
		Bk := 0.

		for j := 0; j <= k; j++ {
			Bk += Psi[j] / math.Pow(M, float64(k-j))
		}

		sum += Bk
		if Bk <= 1e-17*sum {
			break
		}

		k++

		/* equation (68).  Psi_{j} from Psi_{j - 2}.  note that we
		 * have all j < k, we just need to add the j = k-th term to
		 * the sequence */
		if k >= 20 {
			/* FIXME:  allow dynamic allocation of Psi */
			panic("max iteration reached")
		}
		Psi[k] = (float64(k)-1)/M*Psi[k-2] + math.Pow(-zeta_, float64(k)-1)/M*e_to_neg_half_M_zeta_2
	}

	return sf.Erfc(-zeta_*math.Sqrt(M/2.))/2. - math.Sqrt(M/gsl.TwoPi)*sum
}

/*
 * quadrature method in section 5.  the integrand is an even function of
 * theta, so we just integrate over [0, +\pi] and double the result.
 *
 * NOTE:  as theta approaches +\pi, \theta/sin(\theta) diverges in equation
 * (97) causing the cos(\theta)\rho term to go to -\infty and the argument
 * of the logarithm to go to +\infy and thus the logarithm to -\infty;
 * this makes the integrand go to 0 in (95), and so we can back the upper
 * bound away from +\pi a bit without affecting the result and in doing so
 * avoid the need to trap overflow errors inside the argument of the
 * exponential.
 *
 * NOTE:  the expression in (95) only yields Q if \zeta(x, y) < 0,
 * otherwise it yields -P.
 */
type integrand struct {
	M, y, xi float64
}

/*
 * equation (95)
 */
func (i *integrand) Evaluate(theta float64) float64 {
	return math.Exp(i.M*psi(theta, i.xi)) * f(theta, i.y, i.xi)
}

func marcumQQuadrature(M, x, y, xi float64) float64 {
	f := &integrand{M, y, xi}

	/* "limit" must = size of workspace.  upper bound a bit less than
	 * +\pi (see above) */
	workspace, _ := integ.NewWorkspace(20)
	var integral, abserr float64
	integ.Qag(f, 0, math.Pi-1./512., abserr, 1e-12, 20, workspace, &integral, &abserr, integ.Qk41)

	/* instead of dividing by 2*pi and doubling the integral, divide by
	 * pi */
	integral *= math.Exp(-M*halfZeta2(x, y)) / math.Pi

	if x+1. < y {
		return integral
	}

	return 1 + integral
}

//////////////////////////////
//  		PUBLIC  		//
//////////////////////////////

/**
 * The function defined by J.\ Marcum,
 * \f[
 *	Q_{M}(a, b) = \int_{b}^{\infty} x \left( \frac{x}{a} \right)^{M - 1} \exp \left( -\frac{x^{2} + a^{2}}{2} \right) I_{M - 1}(a x) \,\mathrm{d}x,
 * \f]
 * where \f$I_{M - 1}\f$ is the modified Bessel function of order \f$M -
 * 1\f$.
 *
 * The CCDF for the random variable \f$x\f$ distributed according to the
 * noncentral \f$\chi^{2}\f$ distribution with \f$k\f$ degrees-of-freedom
 * and noncentrality parameter \f$\lambda\f$ is \f$Q_{k/2}(\sqrt{\lambda},
 * \sqrt{x})\f$.
 *
 * The CCDF for the random variable \f$x\f$ distributed according to the
 * Rice distribution with noncentrality parameter \f$\nu\f$ and width
 * \f$\sigma\f$ is \f$Q_{1}(\nu/\sigma, x/\sigma)\f$.
 *
 * The probability that a signal that would be seen in a two-phase matched
 * filter with |SNR| \f$\rho_{0}\f$ is seen to have matched filter |SNR|
 * \f$\geq \rho\f$ in stationary Gaussian noise is \f$Q_{1}(\rho_{0},
 * \rho)\f$.
 *
 * This function is implemented by computing the modified form used by Gil
 * <I>et al.</I>,
 *
 *	XLALMarcumQ(M, a, b) = XLALMarcumQmodified(M, a * a / 2., b * b / 2.).
 */
func MarcumQ(M, a, b float64) float64 {
	/*
	 * inverse of equation (5).  note:  all error checking handled
	 * inside XLALMarcumQmodified().
	 */
	return MarcumQModified(M, a*a/2., b*b/2.)
}

/**
 * The modified form of the Marcum Q function. Used by Gil <i>et al.</i> in
 *
 * A. Gil, J. Segura, and N. M. Temme.  Algorithm 939:  Computation of the
 * Marcum Q-Function.  ACM Transactions on Mathematical Software (TOMS),
 * Volume 40 Issue 3, April 2014, Article No. 20. arXiv:1311.0681
 *
 * The relationship between this function and the standard Marcum Q
 * function is
 *
 *	XLALMarcumQmodified(M, x, y) = XLALMarcumQ(M, sqrt(2. * x), sqrt(2. * y)).
 *
 * The function is defined for \f$1 \leq M\f$, \f$0 \leq x\f$, \f$0 \leq
 * y\f$.  Additionally, the implementation here becomes inaccurate when
 * \f$M\f$, \f$x\f$, or \f$y\f$ is \f$\geq 10000\f$.
 */

func MarcumQModified(M, x, y float64) float64 {

	var xi, Q_ float64

	if M < 1. {
		panic(fmt.Sprintf("require 1 <= M: M=%.16f", M))
	}
	if x < 0. {
		panic(fmt.Sprintf("0 <= x: x=%.16f", x))
	}
	if y < 0. {
		panic(fmt.Sprintf("0 <= y: y=%.16f", y))
	}

	if M > 10000. {
		panic(fmt.Sprintf("require M <= 10000: M=%.16f", M))
	}
	if x > 10000. {
		panic(fmt.Sprintf("require x <= 10000: x=%.16f", x))
	}
	if y > 10000. {
		panic(fmt.Sprintf("require y <= 10000: y=%.16f", y))
	}

	xi = 2. * math.Sqrt(x*y)
	f1, f2 := f1f2(x, M)

	/*
	 * selection scheme from section 6
	 */
	if x < 30. {
		/*
		 * use the series expansion in section 3
		 */

		Q_ = marcumQSmallX(M, x, y)
	} else if xi > 30. && M*M < 2.*xi {
		/*
		 * use the asymptotic expansion in section 4.1
		 */

		Q_ = marcumQLargeXY(M, x, y, xi)
	} else if f1 < y && y < f2 && M < 135. {
		/*
		 * use the recurrence relations in (14)
		 */

		Q_ = marcumQRecurrence(M, x, y, xi)
	} else if f1 < y && y < f2 && M >= 135. {
		/*
		 * use the asymptotic expansion in section 4.2
		 */

		/* FIXME:  missing an implementation of f_{jk} needed by
		 * this code path */

		Q_ = marcumQLargeM(M-1., x/(M-1.), y/(M-1.))
	} else {
		/*
		 * use the integral representation in section 5
		 */

		Q_ = marcumQQuadrature(M, x/M, y/M, xi/M)
	}

	/*
	 * we're generally only correct to 12 digits, and the error can
	 * result in an impossible probability.  nudge back into the
	 * allowed range.
	 */
	if 1. < Q_ && Q_ < 1.+1e-12 {
		Q_ = 1.
	}

	if Q_ < 0. || Q_ > 1. {
		panic(fmt.Sprintf("MarcumQmodified(%.v, %.v, %v) = %v", M, x, y, Q_))
	}

	return Q_
}
