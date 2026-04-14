import math
from enum import Enum
from typing import Optional, Dict
from scipy.stats import norm


class CallOrPut(Enum):

    CALL = "call"
    PUT = "put"
    INVALID = "invalid"

    @classmethod
    def from_string(cls, value: str) -> "CallOrPut":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class SimpleMetrics(Enum):

    ## valuations
    PV = "pv"
    ## vol
    IMPLIED_NORMAL_VOL = "implied_normal_vol"
    IMPLIED_LOG_NORMAL_VOL = "implied_log_normal_vol"
    ## pv sensitivities
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    TTE_RISK = "tte_risk"
    STRIKE_RISK = "strike_risk"
    STRIKE_RISK_2 = "strike_risk_2"
    THETA = "theta"

    ## vol sensitivities
    # nv = f(ln_vol, f, k, tte)
    D_N_VOL_D_LN_VOL = "d_n_vol_d_ln_vol"
    D_N_VOL_D_FORWARD = "d_n_vol_d_forward"
    D_N_VOL_D_TTE = "d_n_vol_d_tte"
    D_N_VOL_D_STRIKE = "d_n_vol_d_strike"
    # ln_vol = f^-1(nv, f, k, tte)
    D_LN_VOL_D_N_VOL = "d_ln_vol_d_n_vol"
    D_LN_VOL_D_FORWARD = "d_ln_vol_d_forward"
    D_LN_VOL_D_TTE = "d_ln_vol_d_tte"
    D_LN_VOL_D_STRIKE = "d_ln_vol_d_strike"

    @classmethod
    def from_string(cls, value: str) -> "SimpleMetrics":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class EuropeanOptionAnalytics:

    @staticmethod
    def european_option_log_normal(
        forward: float,
        strike: float,
        time_to_expiry: float,
        log_normal_sigma: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the Black-76 price and analytic Greeks of a European call or put option
        in the forward measure, using lognormal implied volatility.

        res should include
        - SimpleMetrics.PV: present value
        - SimpleMetrics.DELTA: delta
        - SimpleMetrics.GAMMA: gamma
        - SimpleMetrics.VEGA: vega
        - SimpleMetrics.THETA: theta
        - SimpleMetrics.TTE_RISK: time to expiry risk
        - SimpleMetrics.STRIKE_RISK: strike risk

        use calc_risk to control whether to compute the risk metrics or not
        """

        if time_to_expiry <= 0 or log_normal_sigma <= 0:
            raise ValueError("Time to expiry and implied log-normal sigma must be positive")

        res: Dict[SimpleMetrics, float] = {}

        d1 = (math.log(forward / strike) + 0.5 * log_normal_sigma**2 * time_to_expiry) / (
            log_normal_sigma * math.sqrt(time_to_expiry)
        )
        d2 = d1 - log_normal_sigma * math.sqrt(time_to_expiry)

        # pricing
        if option_type == CallOrPut.CALL:
            res[SimpleMetrics.PV] = forward * norm.cdf(d1) - strike * norm.cdf(d2)
        elif option_type == CallOrPut.PUT:
            res[SimpleMetrics.PV] = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be call or put!")

        # risk
        if calc_risk:
            res[SimpleMetrics.DELTA] = norm.cdf(d1) if option_type == CallOrPut.CALL else -norm.cdf(-d1)
            res[SimpleMetrics.GAMMA] = norm.pdf(d1) / (forward * log_normal_sigma * math.sqrt(time_to_expiry))
            res[SimpleMetrics.VEGA] = forward * norm.pdf(d1) * math.sqrt(time_to_expiry)
            res[SimpleMetrics.THETA] = -(forward * norm.pdf(d1) * log_normal_sigma) / (2 * math.sqrt(time_to_expiry))
            res[SimpleMetrics.TTE_RISK] = -res[SimpleMetrics.THETA] 
            res[SimpleMetrics.STRIKE_RISK] = - norm.cdf(d2) if option_type == CallOrPut.CALL else norm.cdf(-d2)

        return res

    @staticmethod
    def european_option_normal(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the Bachelier (normal) price and analytic Greeks of a European call or put option
        in the forward measure, using normal implied volatility.

        res should include
        - SimpleMetrics.PV: present value
        - SimpleMetrics.DELTA: delta
        - SimpleMetrics.GAMMA: gamma
        - SimpleMetrics.VEGA: vega
        - SimpleMetrics.THETA: theta
        - SimpleMetrics.TTE_RISK: time to expiry risk
        - SimpleMetrics.STRIKE_RISK: strike risk

        use calc_risk to control whether to compute the risk metrics or not
        """

        if time_to_expiry <= 0 or normal_sigma <= 0:
            raise ValueError("Time to expiry and implied normal sigma must be positive")

        res: Dict[SimpleMetrics, float] = {}
        
        d = (forward - strike) / (normal_sigma * math.sqrt(time_to_expiry)) 

        # pricing
        if option_type == CallOrPut.CALL:
            res[SimpleMetrics.PV] = (forward - strike) * norm.cdf(d) + normal_sigma * math.sqrt(time_to_expiry) * norm.pdf(d)
        elif option_type == CallOrPut.PUT:
            res[SimpleMetrics.PV] = (strike - forward) * norm.cdf(-d) + normal_sigma * math.sqrt(time_to_expiry) * norm.pdf(d)
        else:
            raise ValueError("Option type must be call or put!")
        # risk
        if calc_risk:
            res[SimpleMetrics.DELTA] = norm.cdf(d) if option_type == CallOrPut.CALL else -norm.cdf(-d)
            res[SimpleMetrics.GAMMA] = norm.pdf(d) / (normal_sigma * math.sqrt(time_to_expiry))
            res[SimpleMetrics.VEGA] = math.sqrt(time_to_expiry) * norm.pdf(d)
            res[SimpleMetrics.THETA] = -0.5 * normal_sigma * norm.pdf(d) / math.sqrt(time_to_expiry)
            res[SimpleMetrics.TTE_RISK] = -res[SimpleMetrics.THETA]
            res[SimpleMetrics.STRIKE_RISK] = -norm.cdf(d) if option_type == CallOrPut.CALL else norm.cdf(-d)


        return res

    @staticmethod
    def implied_lognormal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the implied lognormal volatility from option PV under the Black-76 model and its sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: implied lognormal volatility
        - SimpleMetrics.D_LN_VOL_D_FORWARD: sensitivity of implied lognormal volatility to forward
        - SimpleMetrics.D_LN_VOL_D_TTE: sensitivity of implied lognormal volatility to time to expiry
        - SimpleMetrics.D_LN_VOL_D_STRIKE: sensitivity of implied lognormal volatility to strike

        use calc_risk to control whether to compute the risk metrics or not

        """
        res: Dict[SimpleMetrics, float] = {}

        # 1) compute implied vol
        ln_sigma = EuropeanOptionAnalytics._implied_lognormal_vol_black(pv, forward, strike, time_to_expiry, option_type)
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = ln_sigma 

        # 2) compute greeks at implied vol
        greeks = EuropeanOptionAnalytics.european_option_log_normal(forward, strike, time_to_expiry, ln_sigma, option_type, calc_risk)

        # 3) compute sensitivities of implied vol using implicit function theorem
        # G(\sigma_imp(f, k, tte, pv), f, k, tte) = pv, where G is the pricing function
        # For instance, for f risk, we have
        # dG/dsigma * dsigma / df = - dG/df => - dG/df / dG/dsigma
        if calc_risk:
            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = - greeks[SimpleMetrics.DELTA] / greeks[SimpleMetrics.VEGA] 
            res[SimpleMetrics.D_LN_VOL_D_TTE] = - greeks[SimpleMetrics.TTE_RISK] / greeks[SimpleMetrics.VEGA] 
            res[SimpleMetrics.D_LN_VOL_D_STRIKE] = - greeks[SimpleMetrics.STRIKE_RISK] / greeks[SimpleMetrics.VEGA] 

        return res

    @staticmethod
    def implied_normal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the implied normal volatility from option PV under the Bachelier model and,
        optionally, its sensitivities using the implicit function theorem.

        res should include
        - SimpleMetrics.IMPLIED_NORMAL_VOL: implied normal volatility
        - SimpleMetrics.D_N_VOL_D_FORWARD: sensitivity of implied normal volatility to forward
        - SimpleMetrics.D_N_VOL_D_TTE: sensitivity of implied normal volatility to time to expiry
        - SimpleMetrics.D_N_VOL_D_STRIKE: sensitivity of implied normal volatility to strike

        use calc_risk to control whether to compute the risk metrics or not
        """

        res = {}

        # 1) Compute implied normal vol
        sigma = EuropeanOptionAnalytics._implied_normal_vol_bachelier(pv, forward, strike, time_to_expiry, option_type)
        res[SimpleMetrics.IMPLIED_NORMAL_VOL] = sigma 

        # 2) Compute Greeks at implied vol
        greeks = EuropeanOptionAnalytics.european_option_normal(forward, strike, time_to_expiry, sigma, option_type, calc_risk)

        # 3) Compute sensitivities of implied vol
        # G(\sigma_imp(f, k, tte), f, k, tte) = pv, where G is the pricing function
        # For instance, for f risk, we have
        # dG/dsigma * dsigma / df = - dG/df => - dG/df / dG/dsigma
        if calc_risk:
            res[SimpleMetrics.D_N_VOL_D_FORWARD] = - greeks[SimpleMetrics.DELTA] / greeks[SimpleMetrics.VEGA] 
            res[SimpleMetrics.D_N_VOL_D_TTE] = - greeks[SimpleMetrics.TTE_RISK] / greeks[SimpleMetrics.VEGA] 
            res[SimpleMetrics.D_N_VOL_D_STRIKE] = - greeks[SimpleMetrics.STRIKE_RISK] / greeks[SimpleMetrics.VEGA] 
        return res

    @staticmethod
    def lognormal_vol_to_normal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        log_normal_sigma: float,
        calc_risk: Optional[bool] = False,
        shift: Optional[float] = 0.0,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Converts lognormal implied volatility into normal (Bachelier) implied volatility
        via price equivalence, and compute sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_NORMAL_VOL: equivalent normal implied volatility
        - SimpleMetrics.D_N_VOL_D_LN_VOL: sensitivity of normal vol to lognormal vol
        - SimpleMetrics.D_N_VOL_D_FORWARD: sensitivity of normal vol to forward
        - SimpleMetrics.D_N_VOL_D_STRIKE: sensitivity of normal vol to strike
        - SimpleMetrics.D_N_VOL_D_TTE: sensitivity of normal vol to time to expiry
        """

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) black price (BS'76)
        # V = BS(f, k, tte, log_normal_sigma)
        ln_res = EuropeanOptionAnalytics.european_option_log_normal(
            forward + shift,
            strike + shift,
            time_to_expiry,
            log_normal_sigma,
            option_type,
            calc_risk,
        )
        pv = ln_res[SimpleMetrics.PV]

        # 2) implied normal vol (Bachelier)
        norm_res = EuropeanOptionAnalytics.implied_normal_vol_sensitivities(
            pv,
            forward + shift,
            strike + shift,
            time_to_expiry,
            option_type,
            calc_risk,
            tol
        )
        res[SimpleMetrics.IMPLIED_NORMAL_VOL] = norm_res[SimpleMetrics.IMPLIED_NORMAL_VOL]

        # nv = Imp(f, k, tte, V) where V = BS(f, k, tte, ln_sigma). 
        # dnv/dlnv = dImp/dV * dV/dlnv = BS_vega / bachelier_vega. dImp/dV = 1 / bachelier_vega by implicit function theorem
        # dnv/df = dnv/df (if V didnt depend on f) + dnv/dV * dV/df 
        # Analagous logic for risk wrt strike and tte
        if calc_risk:
            bachelier_vega = EuropeanOptionAnalytics.european_option_normal(
                forward + shift,
                strike + shift,
                time_to_expiry,
                norm_res[SimpleMetrics.IMPLIED_NORMAL_VOL],
                option_type,
                calc_risk,
            )[SimpleMetrics.VEGA]
            res[SimpleMetrics.D_N_VOL_D_LN_VOL] = ln_res[SimpleMetrics.VEGA] / bachelier_vega
            res[SimpleMetrics.D_N_VOL_D_FORWARD] = (
                norm_res[SimpleMetrics.D_N_VOL_D_FORWARD] + ln_res[SimpleMetrics.DELTA] / bachelier_vega
            )
            res[SimpleMetrics.D_N_VOL_D_STRIKE] = (
               norm_res[SimpleMetrics.D_N_VOL_D_STRIKE] + ln_res[SimpleMetrics.STRIKE_RISK] / bachelier_vega
            )
            res[SimpleMetrics.D_N_VOL_D_TTE] = (
               norm_res[SimpleMetrics.D_N_VOL_D_TTE] + ln_res[SimpleMetrics.TTE_RISK] / bachelier_vega
            )

        return res

    @staticmethod
    def normal_vol_to_lognormal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        calc_risk: Optional[bool] = False,
        shift: Optional[float] = 0.0,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Converts normal implied volatility into lognormal implied volatility
        via price equivalence, and computes sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: equivalent lognormal implied volatility
        - SimpleMetrics.D_LN_VOL_D_N_VOL: sensitivity of lognormal vol to normal vol
        - SimpleMetrics.D_LN_VOL_D_FORWARD: sensitivity of lognormal vol to forward
        - SimpleMetrics.D_LN_VOL_D_STRIKE: sensitivity of lognormal vol to strike
        - SimpleMetrics.D_LN_VOL_D_TTE: sensitivity of lognormal vol to time to expiry
        """

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) bachelier
        # V = Bachelier(f, k, tte, normal_sigma)
        norm_res = EuropeanOptionAnalytics.european_option_normal(
            forward + shift,
            strike + shift,
            time_to_expiry,
            normal_sigma,
            option_type,
            calc_risk,
        )
        pv = norm_res[SimpleMetrics.PV]

        # 2) implied log normal vol (BS'76)
        # ln_nv = Imp(f, k, tte, V)
        # notice dln_nv/dV = 1 / vega
        ln_res = EuropeanOptionAnalytics.implied_lognormal_vol_sensitivities(
            pv,
            forward + shift,
            strike + shift,
            time_to_expiry,
            option_type,
            calc_risk,
            tol
        )
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = ln_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]

        # risk
        if calc_risk:
            bs_vega = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift,
                strike + shift,
                time_to_expiry,
                ln_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL],
                option_type,
                calc_risk,
            )[SimpleMetrics.VEGA]
            res[SimpleMetrics.D_LN_VOL_D_N_VOL] = norm_res[SimpleMetrics.VEGA] / bs_vega
            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = (
                ln_res[SimpleMetrics.D_LN_VOL_D_FORWARD] + norm_res[SimpleMetrics.DELTA] / bs_vega
            )
            res[SimpleMetrics.D_LN_VOL_D_STRIKE] = (
               ln_res[SimpleMetrics.D_LN_VOL_D_STRIKE] + norm_res[SimpleMetrics.STRIKE_RISK] / bs_vega
            )
            res[SimpleMetrics.D_LN_VOL_D_TTE] = (
               ln_res[SimpleMetrics.D_LN_VOL_D_TTE] + norm_res[SimpleMetrics.TTE_RISK] / bs_vega
            )

        return res

    ### utilities below

    @staticmethod
    def _implied_lognormal_vol_black(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        tol: Optional[float] = 1e-8,
        vol_min: Optional[float] = 0.0,
        vol_max: Optional[float] = 10.0,
        max_iter: Optional[int] = 1000,
    ) -> float:
        """
        Solves for the Black-76 implied lognormal volatility from a European option price using a
        hybrid Newton-Raphson and bisection method, subject to arbitrage bounds and convergence
        controls.

        Return "sigma" implied lognormal volatility
        """

        # Check that pv > intrinsic value 
        intrinsic_val = max(0.0, forward - strike) if option_type == CallOrPut.CALL else max(0.0, strike - forward)
        if intrinsic_val > pv: 
            raise ValueError("Intrinsic Value is less than pv!")
        
        sigma = EuropeanOptionAnalytics._initial_log_normal_implied_vol_guess(forward, time_to_expiry, pv)

        # Recall newton raphson method where x_{n+1} = x_n - f(x_n)/f'(x_n), f(.) = implied_pv - actual_pv 
        for _ in range(max_iter):
            res = EuropeanOptionAnalytics.european_option_log_normal(forward,strike,time_to_expiry, sigma, option_type, True)
            implied_pv = res[SimpleMetrics.PV]
            implied_vega = res[SimpleMetrics.VEGA]

            diff = implied_pv - pv 
            if abs(diff) < tol:
                return sigma

            # Update based on old value of sigma first for bisection fallback
            if implied_pv > pv:
                vol_max = sigma
            else:
                vol_min = sigma

            # newton step only if stable
            if implied_vega > 1e-8 and vol_min < sigma - diff / implied_vega < vol_max:
                sigma -= diff / implied_vega
            else:
                # bisection fallback
                sigma = 0.5 * (vol_min + vol_max)
        raise ValueError(f"Method did not converge within {max_iter} iterations") 


    @staticmethod
    def _implied_normal_vol_bachelier(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        tol: Optional[float] = 1e-8,
        vol_min: Optional[float] = 1e-8,
        vol_max: Optional[float] = 0.1,
        max_iter: Optional[int] = 100,
    ) -> float:
        """
        Solves for the Bachelier implied normal volatility from a European option price using a
        hybrid Newton-Raphson and bisection method, subject to arbitrage bounds and convergence
        controls.

        Return "sigma" implied lognormal volatility
        """
        intrinsic_val = max(0.0, forward - strike) if option_type == CallOrPut.CALL else max(0.0, strike - forward)
        if intrinsic_val > pv: 
            raise ValueError("Intrinsic Value is less than pv!")
        
        sigma = EuropeanOptionAnalytics._initial_normal_implied_vol_guess(time_to_expiry, pv)

        # Recall newton raphson method where x_{n+1} = x_n - f(x_n)/f'(x_n), f(.) = implied_pv - actual_pv 
        for _ in range(max_iter):
            res = EuropeanOptionAnalytics.european_option_normal(forward,strike,time_to_expiry, sigma, option_type, True)
            implied_pv = res[SimpleMetrics.PV]
            implied_vega = res[SimpleMetrics.VEGA]

            diff = implied_pv - pv 
            if abs(diff) < tol:
                return sigma

            # Update based on old value of sigma first for bisection fallback
            if implied_pv > pv:
                vol_max = sigma
            else:
                vol_min = sigma

            # newton step only if stable
            if implied_vega > 1e-8 and vol_min < sigma - diff / implied_vega < vol_max:
                sigma -= diff / implied_vega
            else:
                # bisection fallback
                sigma = 0.5 * (vol_min + vol_max)
        raise ValueError(f"Method did not converge within {max_iter} iterations") 
    
    @staticmethod
    def _initial_log_normal_implied_vol_guess(forward: float, time_to_expiry: float, pv: float):
        return math.sqrt(2 * math.pi / time_to_expiry) * pv / forward

    @staticmethod
    def _initial_normal_implied_vol_guess(time_to_expiry: float, pv: float):
        return pv * math.sqrt(2 * math.pi / time_to_expiry)
