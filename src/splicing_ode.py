import numpy as np
from scipy.integrate import solve_ivp


class AlternativeSplicingModel:
    """
    5-state mechanistic co-transcriptional alternative splicing model featuring:
      - Multi-exon sequential commitment states (P1 -> P2 -> P3 -> M_inc vs. M_skip)
      - Non-linear cooperative activator/repressor Hill binding kinetics
      - Isoform-specific mRNA degradation rates (k_deg_inc != k_deg_skip, e.g., NMD)
      - Exact closed-form steady-state PSI* equation and O(1) analytical SF inversion
    """

    def __init__(
        self,
        k_tx: float = 1.0,
        k_commit: float = 2.5,
        k_splice_0: float = 2.0,
        k_skip: float = 0.5,
        k_mat: float = 1.5,
        k_deg_inc: float = 0.2,
        k_deg_skip: float = 0.2,
        K_act: float = 1.0,
        n_act: float = 2.0,
        repressor: float = 0.0,
        K_rep: float = 1.0,
        n_rep: float = 2.0,
    ) -> None:
        self.k_tx = k_tx
        self.k_commit = k_commit
        self.k_splice_0 = k_splice_0
        self.k_skip = k_skip
        self.k_mat = k_mat
        self.k_deg_inc = k_deg_inc
        self.k_deg_skip = k_deg_skip
        self.K_act = K_act
        self.n_act = n_act
        self.repressor = repressor
        self.K_rep = K_rep
        self.n_rep = n_rep

    def effective_splice_rate(self, sf_activity: float) -> float:
        """Non-linear cooperative binding rate k_splice(SF, repressor)."""
        act_term = (max(sf_activity, 0.0) / self.K_act) ** self.n_act
        rep_term = (max(self.repressor, 0.0) / self.K_rep) ** self.n_rep
        return self.k_splice_0 * (act_term / (1.0 + act_term + rep_term))

    def steady_state(self, sf_activity: float = 1.0) -> dict[str, float]:
        """Exact closed-form equilibrium abundances and steady-state PSI*."""
        k_sp = self.effective_splice_rate(sf_activity)
        p1_ss = self.k_tx / self.k_commit
        p2_ss = self.k_tx / (k_sp + self.k_skip)
        p3_ss = (k_sp / self.k_mat) * p2_ss
        m_inc_ss = (k_sp / self.k_deg_inc) * p2_ss
        m_skip_ss = (self.k_skip / self.k_deg_skip) * p2_ss
        psi_ss = m_inc_ss / (m_inc_ss + m_skip_ss)
        return {
            "P1": p1_ss,
            "P2": p2_ss,
            "P3": p3_ss,
            "M_inc": m_inc_ss,
            "M_skip": m_skip_ss,
            "PSI": psi_ss,
        }

    def infer_sf_closed_form(self, psi_obs: float, bounds: tuple[float, float] = (0.01, 5.0)) -> float:
        """O(1) exact analytical inversion of the closed-form steady-state PSI* equation."""
        psi_c = float(np.clip(psi_obs, 1e-4, 1.0 - 1e-4))
        k_req = self.k_skip * (self.k_deg_inc / self.k_deg_skip) * (psi_c / (1.0 - psi_c))
        if k_req >= self.k_splice_0 * 0.999:
            return bounds[1]
        rep_term = (max(self.repressor, 0.0) / self.K_rep) ** self.n_rep
        act_ratio = (k_req * (1.0 + rep_term)) / (self.k_splice_0 - k_req)
        sf_est = self.K_act * (max(act_ratio, 0.0) ** (1.0 / self.n_act))
        return float(np.clip(sf_est, bounds[0], bounds[1]))

    def odes(self, t: float, y: np.ndarray, sf_activity: float = 1.0) -> list[float]:
        """5-state coupled ODE derivatives [dP1/dt, dP2/dt, dP3/dt, dM_inc/dt, dM_skip/dt]."""
        p1, p2, p3, m_inc, m_skip = y
        k_sp = self.effective_splice_rate(sf_activity)

        dp1 = self.k_tx - self.k_commit * p1
        dp2 = self.k_commit * p1 - (k_sp + self.k_skip) * p2
        dp3 = k_sp * p2 - self.k_mat * p3
        dm_inc = self.k_mat * p3 - self.k_deg_inc * m_inc
        dm_skip = self.k_skip * p2 - self.k_deg_skip * m_skip
        return [dp1, dp2, dp3, dm_inc, dm_skip]

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 120.0),
        n_points: int = 200,
        sf_activity: float = 1.0,
        y0: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Integrate the 5-state ODE system using LSODA and return trajectories + closed-form PSI*."""
        if y0 is None:
            y0 = [0.0, 0.0, 0.0, 0.0, 0.0]
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            fun=lambda t, y: self.odes(t, y, sf_activity=sf_activity),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="LSODA",
        )
        p1, p2, p3, m_inc, m_skip = sol.y
        psi = m_inc / (m_inc + m_skip + 1e-12)
        return {
            "time": sol.t,
            "P1": p1,
            "P2": p2,
            "P3": p3,
            "M_inc": m_inc,
            "M_skip": m_skip,
            "PSI": psi,
            "PSI_steady_state": self.steady_state(sf_activity)["PSI"],
        }
