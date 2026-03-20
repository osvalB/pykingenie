import pytest
import numpy as np

from pykingenie.utils.signal_surface import (
    solve_two_site_association,
    solve_two_site_cooperative_association,
    solve_two_site_dissociation,
    solve_two_site_cooperative_dissociation,
    steady_state_two_site,
    steady_state_two_site_cooperative,
)


class TestSolveTwoSiteAssociation:
    """Tests for the non-cooperative two-site association solver."""

    def test_high_concentration_signal_equals_Rmax_LPL(self):
        """At extremely high analyte concentration all ligand becomes doubly
        bound (fLPL → 1, fPL → 0), so the total signal must equal Rmax_LPL."""
        Rmax_PL  = 5.0
        Rmax_LPL = 10.0
        kon  = 1e5
        koff = 0.01

        # Very high concentration drives both sites to full occupancy
        a_conc = 1e6

        time = np.linspace(0, 500, 300)
        result = solve_two_site_association(
            time, a_conc, kon, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )

        total_signal = result[-1, 0]
        signal_PL    = result[-1, 1]
        signal_LPL   = result[-1, 2]

        # Total signal should converge to Rmax_LPL
        assert np.isclose(total_signal, Rmax_LPL, rtol=1e-3), (
            f"At very high [L], total signal ({total_signal:.6f}) "
            f"should equal Rmax_LPL ({Rmax_LPL})"
        )

        # PL component should vanish
        assert np.isclose(signal_PL, 0, atol=1e-3), (
            f"At very high [L], PL signal ({signal_PL:.6f}) should be ~0"
        )

        # LPL component should equal Rmax_LPL
        assert np.isclose(signal_LPL, Rmax_LPL, rtol=1e-3), (
            f"At very high [L], LPL signal ({signal_LPL:.6f}) "
            f"should equal Rmax_LPL ({Rmax_LPL})"
        )

    def test_returns_correct_shape(self):
        """Output must have shape (len(time), 3)."""
        time = np.linspace(0, 50, 100)
        result = solve_two_site_association(
            time, a_conc=1.0, kon=0.1, koff=0.01,
            Rmax_PL=5.0, Rmax_LPL=10.0,
        )
        assert result.shape == (100, 3)

    def test_signal_starts_at_zero(self):
        """With default initial fractions (0, 0) the signal must start at 0."""
        time = np.linspace(0, 50, 100)
        result = solve_two_site_association(
            time, a_conc=1.0, kon=0.1, koff=0.01,
            Rmax_PL=5.0, Rmax_LPL=10.0,
        )
        assert np.isclose(result[0, 0], 0, atol=1e-10)

    def test_signal_is_non_decreasing(self):
        """During association from an empty sensor the total signal must not
        decrease."""
        time = np.linspace(0, 200, 500)
        result = solve_two_site_association(
            time, a_conc=1.0, kon=0.1, koff=0.01,
            Rmax_PL=5.0, Rmax_LPL=10.0,
        )
        total = result[:, 0]
        assert np.all(np.diff(total) >= -1e-10), "Total signal should be non-decreasing"

    def test_conservation_of_fractions(self):
        """fP + fPL + fLPL must equal 1 at every time point."""
        time = np.linspace(0, 100, 200)
        # Use Rmax=1 so that signals equal fractions directly
        result = solve_two_site_association(
            time, a_conc=1.0, kon=0.1, koff=0.01,
            Rmax_PL=1.0, Rmax_LPL=1.0,
        )
        fPL  = result[:, 1]
        fLPL = result[:, 2]
        fP   = 1 - fPL - fLPL

        assert np.all(fP >= -1e-10), "Free fraction must be non-negative"
        assert np.all(fP <= 1 + 1e-10), "Free fraction must be at most 1"

        # fractions must sum one
        np.testing.assert_allclose(fP + fPL + fLPL, 1, atol=1e-10)

class TestSteadyStateAgreement:
    """The kinetic solver run to long times must converge to the analytical
    steady-state formula."""

    def test_association_matches_steady_state(self):
        kon, koff = 0.1, 0.01
        Kd = koff / kon
        Rmax_PL, Rmax_LPL = 5.0, 10.0

        for a_conc in [0.01, 0.1, 1.0, 10.0]:
            time = np.linspace(0, 5000, 500)
            result = solve_two_site_association(
                time, a_conc, kon, koff,
                Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            )
            expected = steady_state_two_site(
                a_conc, Rmax_PL, Rmax_LPL, Kd,
            )
            assert np.isclose(result[-1, 0], expected, rtol=1e-3), (
                f"At a_conc={a_conc}, kinetic steady state ({result[-1, 0]:.6f}) "
                f"should match analytical ({expected:.6f})"
            )

    def test_cooperative_association_matches_steady_state(self):
        kon, koff, sigma = 0.1, 0.01, 5.0
        Kd = koff / kon
        Rmax_PL, Rmax_LPL = 5.0, 10.0

        for a_conc in [0.01, 0.1, 1.0, 10.0]:
            time = np.linspace(0, 5000, 500)
            result = solve_two_site_cooperative_association(
                time, a_conc, kon, koff, sigma,
                Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            )
            expected = steady_state_two_site_cooperative(
                a_conc, Rmax_PL, Rmax_LPL, Kd, sigma,
            )
            assert np.isclose(result[-1, 0], expected, rtol=1e-3), (
                f"Cooperative (σ={sigma}) at a_conc={a_conc}: kinetic "
                f"({result[-1, 0]:.6f}) != analytical ({expected:.6f})"
            )

    def test_strong_negative_cooperativity_high_ligand_matches_steady_state(self):
        """At very high ligand and long time, strong negative cooperativity
        must converge to the cooperative analytical steady state (near Rmax_PL)."""
        kon, koff = 0.1, 0.01
        sigma = 1e-10
        Kd = koff / kon
        a_conc = 100.0
        Rmax_PL, Rmax_LPL = 1, 10.0

        time = np.linspace(0, 5000, 500)
        result = solve_two_site_cooperative_association(
            time, a_conc, kon, koff, sigma,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )

        kinetic_ss = result[-1, 0]
        expected_ss = steady_state_two_site_cooperative(
            a_conc, Rmax_PL, Rmax_LPL, Kd, sigma,
        )

        assert np.isclose(kinetic_ss, expected_ss, rtol=1e-3), (
            f"Strong negative cooperativity at high [L]: kinetic "
            f"({kinetic_ss:.6f}) != analytical ({expected_ss:.6f})"
        )
        assert np.isclose(kinetic_ss, Rmax_PL, rtol=1e-2), (
            f"Strong negative cooperativity at high [L]: signal "
            f"({kinetic_ss:.6f}) should be close to Rmax_PL ({Rmax_PL})"
        )


class TestAnalyticalDissociation:
    """The dissociation ODE has a closed-form solution (triangular matrix).
    Compare the solver output to the exact formulas:

        fLPL(t) = fLPL_0 · exp(−2·koff·t)
        fPL(t)  = (fPL_0 + 2·fLPL_0)·exp(−koff·t)
                  − 2·fLPL_0·exp(−2·koff·t)
    """

    def test_dissociation_matches_analytical(self):
        koff = 0.05
        fPL_0, fLPL_0 = 0.3, 0.5
        Rmax_PL, Rmax_LPL = 1.0, 1.0  # so signal == fraction

        time = np.linspace(0, 100, 500)
        result = solve_two_site_dissociation(
            time, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            fPL_0=fPL_0, fLPL_0=fLPL_0,
        )

        t = time - time[0]
        fLPL_exact = fLPL_0 * np.exp(-2 * koff * t)
        fPL_exact = (
            (fPL_0 + 2 * fLPL_0) * np.exp(-koff * t)
            - 2 * fLPL_0 * np.exp(-2 * koff * t)
        )

        np.testing.assert_allclose(result[:, 1], fPL_exact, atol=1e-8)
        np.testing.assert_allclose(result[:, 2], fLPL_exact, atol=1e-8)

    def test_dissociation_decays_to_zero(self):
        """After long enough dissociation all signal must vanish."""
        koff = 0.1
        fPL_0, fLPL_0 = 0.4, 0.4

        time = np.linspace(0, 500, 300)
        result = solve_two_site_dissociation(
            time, koff,
            Rmax_PL=5.0, Rmax_LPL=10.0,
            fPL_0=fPL_0, fLPL_0=fLPL_0,
        )
        assert np.isclose(result[-1, 0], 0, atol=1e-6), (
            f"Dissociation signal should decay to 0, got {result[-1, 0]:.6f}"
        )


class TestCooperativity:
    """Verify that the cooperativity factor sigma behaves as expected."""

    def test_sigma_one_matches_non_cooperative_association(self):
        """With sigma=1 the cooperative solver must reproduce the
        non-cooperative result exactly."""
        time = np.linspace(0, 200, 300)
        kwargs = dict(
            a_conc=1.0, kon=0.1, koff=0.01,
            Rmax_PL=5.0, Rmax_LPL=10.0,
        )
        ref = solve_two_site_association(time, **kwargs)
        coop = solve_two_site_cooperative_association(time, sigma=1.0, **kwargs)
        np.testing.assert_allclose(coop, ref, atol=1e-10)

    def test_sigma_one_matches_non_cooperative_dissociation(self):
        time = np.linspace(0, 200, 300)
        kwargs = dict(
            koff=0.05,
            Rmax_PL=5.0, Rmax_LPL=10.0,
            fPL_0=0.3, fLPL_0=0.5,
        )
        ref = solve_two_site_dissociation(time, **kwargs)
        coop = solve_two_site_cooperative_dissociation(time, sigma=1.0, **kwargs)
        np.testing.assert_allclose(coop, ref, atol=1e-10)

    def test_positive_cooperativity_increases_LPL(self):
        """sigma > 1 should yield more LPL at steady state than
        the non-cooperative case at the same concentration."""
        kon, koff = 0.1, 0.01
        a_conc = 1.0
        Rmax_PL, Rmax_LPL = 1.0, 1.0
        time = np.linspace(0, 5000, 500)

        ref = solve_two_site_association(
            time, a_conc, kon, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )
        coop = solve_two_site_cooperative_association(
            time, a_conc, kon, koff, sigma=10.0,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )
        assert coop[-1, 2] > ref[-1, 2], (
            "Positive cooperativity should increase LPL fraction"
        )

    def test_negative_cooperativity_decreases_LPL(self):
        """sigma < 1 should yield less LPL at steady state."""
        kon, koff = 0.1, 0.01
        a_conc = 1.0
        Rmax_PL, Rmax_LPL = 1.0, 1.0
        time = np.linspace(0, 5000, 500)

        ref = solve_two_site_association(
            time, a_conc, kon, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )
        coop = solve_two_site_cooperative_association(
            time, a_conc, kon, koff, sigma=0.1,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )
        assert coop[-1, 2] < ref[-1, 2], (
            "Negative cooperativity should decrease LPL fraction"
        )

    def test_strong_negative_cooperativity_signal_stays_at_Rmax_PL(self):
        """With very strong negative cooperativity (sigma ≈ 0) the second
        binding step is essentially blocked.  At high analyte concentration
        all ligand becomes singly-bound (PL) but NOT doubly-bound (LPL),
        so the total signal must converge to Rmax_PL, not Rmax_LPL."""
        kon, koff = 0.1, 0.01
        Rmax_PL  = 5.0
        Rmax_LPL = 10.0
        sigma    = 1e-10          # extremely strong negative cooperativity
        a_conc   = 100.0          # high concentration so all sites try to bind

        time = np.linspace(0, 5000, 500)
        result = solve_two_site_cooperative_association(
            time, a_conc, kon, koff, sigma,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )

        total_signal = result[-1, 0]
        signal_PL    = result[-1, 1]
        signal_LPL   = result[-1, 2]

        # Total signal should be close to Rmax_PL (singly-bound dominates)
        assert np.isclose(total_signal, Rmax_PL, rtol=1e-2), (
            f"With very strong negative cooperativity the total signal "
            f"({total_signal:.6f}) should converge to Rmax_PL ({Rmax_PL})"
        )

        # LPL signal should be negligible
        assert signal_LPL < 0.01 * Rmax_LPL, (
            f"LPL signal ({signal_LPL:.6f}) should be negligible"
        )

        # PL signal should dominate
        assert np.isclose(signal_PL, Rmax_PL, rtol=1e-2), (
            f"PL signal ({signal_PL:.6f}) should be close to Rmax_PL ({Rmax_PL})"
        )

    def test_cooperative_dissociation_slower_with_high_sigma(self):
        """sigma > 1 makes the second off-rate koff/sigma, so LPL should
        dissociate more slowly than in the non-cooperative case."""
        koff = 0.1
        fPL_0, fLPL_0 = 0.1, 0.8
        time = np.linspace(0, 50, 300)

        ref = solve_two_site_dissociation(
            time, koff,
            Rmax_PL=1.0, Rmax_LPL=1.0,
            fPL_0=fPL_0, fLPL_0=fLPL_0,
        )
        coop = solve_two_site_cooperative_dissociation(
            time, koff, sigma=10.0,
            Rmax_PL=1.0, Rmax_LPL=1.0,
            fPL_0=fPL_0, fLPL_0=fLPL_0,
        )
        # At a mid-time point, cooperative total signal should be higher
        # (slower decay)
        mid = len(time) // 2
        assert coop[mid, 0] > ref[mid, 0], (
            "Cooperative dissociation (σ>1) should be slower"
        )


class TestAssociationDissociationContinuity:
    """The signal at the start of dissociation must equal the signal at the
    end of association (no discontinuous jump)."""

    def test_continuity_non_cooperative(self):
        kon, koff = 0.1, 0.01
        Rmax_PL, Rmax_LPL = 5.0, 10.0

        t_assoc = np.linspace(0, 300, 500)
        assoc = solve_two_site_association(
            t_assoc, a_conc=1.0, kon=kon, koff=koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )

        fPL_end  = assoc[-1, 1] / Rmax_PL
        fLPL_end = assoc[-1, 2] / Rmax_LPL

        t_diss = np.linspace(0, 300, 500)
        diss = solve_two_site_dissociation(
            t_diss, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            fPL_0=fPL_end, fLPL_0=fLPL_end,
        )

        np.testing.assert_allclose(
            assoc[-1], diss[0], atol=1e-6,
            err_msg="Dissociation must start where association ended",
        )

    def test_continuity_cooperative(self):
        kon, koff, sigma = 0.1, 0.01, 5.0
        Rmax_PL, Rmax_LPL = 5.0, 10.0

        t_assoc = np.linspace(0, 300, 500)
        assoc = solve_two_site_cooperative_association(
            t_assoc, a_conc=1.0, kon=kon, koff=koff, sigma=sigma,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )

        fPL_end  = assoc[-1, 1] / Rmax_PL
        fLPL_end = assoc[-1, 2] / Rmax_LPL

        t_diss = np.linspace(0, 300, 500)
        diss = solve_two_site_cooperative_dissociation(
            t_diss, koff, sigma,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            fPL_0=fPL_end, fLPL_0=fLPL_end,
        )

        np.testing.assert_allclose(
            assoc[-1], diss[0], atol=1e-6,
            err_msg="Cooperative dissociation must start where association ended",
        )


class TestLowConcentrationLimit:
    """At very low analyte concentration almost no binding occurs; the tiny
    amount that does bind is overwhelmingly singly-bound (PL >> LPL)."""

    def test_low_concentration_mostly_PL(self):
        kon, koff = 0.1, 0.01
        a_conc = 1e-6  # very dilute
        Rmax_PL, Rmax_LPL = 1.0, 1.0

        time = np.linspace(0, 5000, 500)
        result = solve_two_site_association(
            time, a_conc, kon, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
        )
        fPL_end  = result[-1, 1]
        fLPL_end = result[-1, 2]

        # LPL should be negligible compared to PL
        assert fLPL_end < fPL_end * 1e-3, (
            f"At very low [L], fLPL ({fLPL_end:.2e}) should be negligible "
            f"compared to fPL ({fPL_end:.2e})"
        )


class TestColumnConsistency:
    """Column 0 (total) must equal column 1 (PL) + column 2 (LPL) at every
    time point, for every solver."""

    @pytest.mark.parametrize("solver, extra_kwargs", [
        (solve_two_site_association,
         dict(a_conc=1.0, kon=0.1, koff=0.01, Rmax_PL=5.0, Rmax_LPL=10.0)),
        (solve_two_site_cooperative_association,
         dict(a_conc=1.0, kon=0.1, koff=0.01, sigma=3.0, Rmax_PL=5.0, Rmax_LPL=10.0)),
        (solve_two_site_dissociation,
         dict(koff=0.05, Rmax_PL=5.0, Rmax_LPL=10.0, fPL_0=0.3, fLPL_0=0.5)),
        (solve_two_site_cooperative_dissociation,
         dict(koff=0.05, sigma=3.0, Rmax_PL=5.0, Rmax_LPL=10.0, fPL_0=0.3, fLPL_0=0.5)),
    ])
    def test_total_equals_sum_of_components(self, solver, extra_kwargs):
        time = np.linspace(0, 200, 300)
        result = solver(time, **extra_kwargs)
        np.testing.assert_allclose(
            result[:, 0], result[:, 1] + result[:, 2], atol=1e-12,
            err_msg=f"total ≠ PL + LPL for {solver.__name__}",
        )


class TestKnownValueAtKd:
    """At C = Kd the single-site occupancy θ = 0.5, so for identical
    independent sites:  fPL = 2·θ·(1−θ) = 0.5,  fLPL = θ² = 0.25."""

    def test_fractions_at_Kd(self):
        kon, koff = 0.1, 0.01
        Kd = koff / kon
        a_conc = Kd  # θ = 0.5

        time = np.linspace(0, 10000, 500)
        result = solve_two_site_association(
            time, a_conc, kon, koff,
            Rmax_PL=1.0, Rmax_LPL=1.0,  # signal == fraction
        )

        fPL_ss  = result[-1, 1]
        fLPL_ss = result[-1, 2]

        assert np.isclose(fPL_ss, 0.5, rtol=1e-3), (
            f"At C=Kd, fPL should be 0.5, got {fPL_ss:.6f}"
        )
        assert np.isclose(fLPL_ss, 0.25, rtol=1e-3), (
            f"At C=Kd, fLPL should be 0.25, got {fLPL_ss:.6f}"
        )

    def test_steady_state_individual_fractions(self):
        """Verify the analytical steady-state returns the correct individual
        fractions (fPL and fLPL), not just the total signal."""
        Kd = 0.1
        for C in [0.01, 0.1, 1.0, 10.0]:
            theta = C / (C + Kd)
            expected_fPL  = 2 * theta * (1 - theta)
            expected_fLPL = theta ** 2

            # Rmax_PL=1, Rmax_LPL=0 isolates the PL contribution
            signal_PL_only = steady_state_two_site(C, Rmax_PL=1.0, Rmax_LPL=0.0, Kd=Kd)
            assert np.isclose(signal_PL_only, expected_fPL, rtol=1e-10), (
                f"fPL mismatch at C={C}"
            )

            # Rmax_PL=0, Rmax_LPL=1 isolates the LPL contribution
            signal_LPL_only = steady_state_two_site(C, Rmax_PL=0.0, Rmax_LPL=1.0, Kd=Kd)
            assert np.isclose(signal_LPL_only, expected_fLPL, rtol=1e-10), (
                f"fLPL mismatch at C={C}"
            )


class TestCooperativeAnalyticalDissociation:
    r"""Closed-form for cooperative dissociation (triangular matrix).

    Let k₂ = koff / √σ.  Then:

        fLPL(t) = fLPL₀ · exp(−2·k₂·t)
        fPL(t)  = (fPL₀ − α)·exp(−koff·t) + α·exp(−2·k₂·t)

    where  α = 2·k₂·fLPL₀ / (koff − 2·k₂).
    """

    @pytest.mark.parametrize("sigma", [0.25, 2.0, 10.0])
    def test_cooperative_dissociation_matches_analytical(self, sigma):
        koff = 0.05
        fPL_0, fLPL_0 = 0.3, 0.5
        k2 = koff / np.sqrt(sigma)

        time = np.linspace(0, 120, 600)
        result = solve_two_site_cooperative_dissociation(
            time, koff, sigma,
            Rmax_PL=1.0, Rmax_LPL=1.0,
            fPL_0=fPL_0, fLPL_0=fLPL_0,
        )

        t = time - time[0]
        fLPL_exact = fLPL_0 * np.exp(-2 * k2 * t)

        alpha = 2 * k2 * fLPL_0 / (koff - 2 * k2)
        fPL_exact = (fPL_0 - alpha) * np.exp(-koff * t) + alpha * np.exp(-2 * k2 * t)

        np.testing.assert_allclose(result[:, 2], fLPL_exact, atol=1e-8,
                                   err_msg=f"fLPL mismatch for σ={sigma}")
        np.testing.assert_allclose(result[:, 1], fPL_exact, atol=1e-8,
                                   err_msg=f"fPL mismatch for σ={sigma}")


class TestZeroConcentration:
    """With zero analyte concentration the association solver must not change
    the signal from its initial value."""

    def test_no_binding_at_zero_concentration(self):
        time = np.linspace(0, 500, 300)
        result = solve_two_site_association(
            time, a_conc=0.0, kon=0.1, koff=0.01,
            Rmax_PL=5.0, Rmax_LPL=10.0,
            fPL_0=0.0, fLPL_0=0.0,
        )
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-12)

    def test_preloaded_dissociates_at_zero_concentration(self):
        """If sensor is pre-loaded and analyte concentration is zero, the
        association solver should behave like dissociation (signal decays)."""
        kon, koff = 0.1, 0.01
        fPL_0, fLPL_0 = 0.3, 0.4
        Rmax_PL, Rmax_LPL = 1.0, 1.0

        time = np.linspace(0, 5000, 500)
        assoc_zero = solve_two_site_association(
            time, a_conc=0.0, kon=kon, koff=koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            fPL_0=fPL_0, fLPL_0=fLPL_0,
        )
        diss = solve_two_site_dissociation(
            time, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            fPL_0=fPL_0, fLPL_0=fLPL_0,
        )
        np.testing.assert_allclose(assoc_zero, diss, atol=1e-8,
                                   err_msg="Association at [L]=0 should equal dissociation")


class TestDissociationConservation:
    """fP + fPL + fLPL = 1 must hold during dissociation as well."""

    @pytest.mark.parametrize("sigma", [None, 0.5, 5.0])
    def test_conservation_during_dissociation(self, sigma):
        koff = 0.05
        fPL_0, fLPL_0 = 0.3, 0.5

        time = np.linspace(0, 100, 400)
        if sigma is None:
            result = solve_two_site_dissociation(
                time, koff, Rmax_PL=1.0, Rmax_LPL=1.0,
                fPL_0=fPL_0, fLPL_0=fLPL_0,
            )
        else:
            result = solve_two_site_cooperative_dissociation(
                time, koff, sigma, Rmax_PL=1.0, Rmax_LPL=1.0,
                fPL_0=fPL_0, fLPL_0=fLPL_0,
            )

        fPL  = result[:, 1]
        fLPL = result[:, 2]
        fP   = 1 - fPL - fLPL

        assert np.all(fP >= -1e-10), "Free fraction must be non-negative"
        assert np.all(fPL >= -1e-10), "PL fraction must be non-negative"
        assert np.all(fLPL >= -1e-10), "LPL fraction must be non-negative"
        np.testing.assert_allclose(fP + fPL + fLPL, 1.0, atol=1e-10)


class TestConvergenceFromDifferentInitialConditions:
    """Regardless of the starting fractions, the association solver must
    converge to the same steady state."""

    def test_same_steady_state(self):
        kon, koff = 0.1, 0.01
        a_conc = 1.0
        Rmax_PL, Rmax_LPL = 5.0, 10.0

        time = np.linspace(0, 10000, 500)

        # Start from empty sensor
        r1 = solve_two_site_association(
            time, a_conc, kon, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            fPL_0=0.0, fLPL_0=0.0,
        )
        # Start from a pre-loaded sensor
        r2 = solve_two_site_association(
            time, a_conc, kon, koff,
            Rmax_PL=Rmax_PL, Rmax_LPL=Rmax_LPL,
            fPL_0=0.8, fLPL_0=0.1,
        )
        np.testing.assert_allclose(
            r1[-1], r2[-1], rtol=1e-3,
            err_msg="Different initial conditions must converge to the same steady state",
        )


class TestRmaxLinearity:
    """The signal must scale linearly with the Rmax values."""

    def test_doubling_rmax_doubles_signal(self):
        time = np.linspace(0, 200, 300)
        kwargs = dict(a_conc=1.0, kon=0.1, koff=0.01)

        r1 = solve_two_site_association(
            time, Rmax_PL=3.0, Rmax_LPL=7.0, **kwargs,
        )
        r2 = solve_two_site_association(
            time, Rmax_PL=6.0, Rmax_LPL=14.0, **kwargs,
        )
        np.testing.assert_allclose(r2, 2 * r1, atol=1e-12)

    def test_zero_rmax_gives_zero_signal(self):
        time = np.linspace(0, 200, 300)
        result = solve_two_site_association(
            time, a_conc=1.0, kon=0.1, koff=0.01,
            Rmax_PL=0.0, Rmax_LPL=0.0,
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-15)


