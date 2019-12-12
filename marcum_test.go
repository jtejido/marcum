package marcum_test

import (
	"github.com/jtejido/marcum"
	"math"
	"testing"
)

func TestMarcumQ(t *testing.T) {

	// test data copied from testmarq.f90 in reference implementation by Gil et
	// al. published as part of
	//
	// A. Gil, J. Segura, and N. M. Temme.  Algorithm 939:  Computation of the
	// Marcum Q-Function.  ACM Transactions on Mathematical Software (TOMS),
	// Volume 40 Issue 3, April 2014, Article No. 20. arXiv:1311.0681
	tol := 1e-12

	mu := []float64{1.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, 22.0, 25.0, 27.0, 30.0, 32.0, 40.0, 50.0, 200.0, 350.0, 570.0, 1000.0}

	x := []float64{0.3, 2.0, 8.0, 25.0, 13.0, 45.0, 47.0, 100.0, 85.0, 120.0, 130.0, 140.0, 30.0, 40.0, 0.01, 100.0, 1.0, 0.08}

	y := []float64{0.01, 0.1, 50.0, 10.0, 15.0, 25.0, 30.0, 150.0, 60.0, 205.0, 90.0, 100.0, 120.0, 150.0, 190.0, 320.0, 480.0, 799.0}

	q := []float64{.9926176915580, .9999780077720, .2311934913546e-07, .9998253130004, .8516869957363, .9998251671677, .9999865923082, .3534087845586e-01, .9999821600833, .5457593568564e-03, .9999987797684, .9999982425123, .1052462813144e-04, .3165262228904e-05, .7568702241292, .9999999996149, .9999701550685, .9999999999958}

	for i := 0; i < len(q); i++ {
		res := marcum.MarcumQModified(mu[i], x[i], y[i])
		if math.Abs(q[i]-res)/q[i] >= tol {
			t.Errorf("Mismatch. Case %d, want: %v, got: %v", i, q[i], res)
		}
	}
}
