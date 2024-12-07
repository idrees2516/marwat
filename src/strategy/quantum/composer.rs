use super::types::*;
use ethers::types::U256;

pub struct StrategyComposer {
    market_data: MarketData,
}

impl StrategyComposer {
    pub fn new(market_data: MarketData) -> Self {
        Self { market_data }
    }

    pub fn create_butterfly_spread(
        &self,
        center_strike: U256,
        wing_width: U256,
        size: U256,
        maturity: U256,
    ) -> LeggedPosition {
        let lower_strike = center_strike.saturating_sub(wing_width);
        let upper_strike = center_strike.saturating_add(wing_width);

        let legs = vec![
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: lower_strike,
                    maturity,
                    is_call: true,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: center_strike,
                    maturity,
                    is_call: true,
                    is_long: false,
                    size: size * 2,
                    collateral: size * 2,
                },
                ratio: -2,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: upper_strike,
                    maturity,
                    is_call: true,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
        ];

        LeggedPosition {
            legs,
            total_collateral: size * 4,
            strategy_type: StrategyType::ButterflySpread,
            risk_params: RiskParameters {
                max_loss: size,
                max_leverage: 4.0,
                min_collateral_ratio: 0.25,
                greek_limits: GreekLimits {
                    max_delta: 0.3,
                    max_gamma: 0.1,
                    max_vega: 0.2,
                    max_theta: 0.1,
                },
            },
        }
    }

    pub fn create_iron_condor(
        &self,
        put_center: U256,
        call_center: U256,
        wing_width: U256,
        size: U256,
        maturity: U256,
    ) -> LeggedPosition {
        let put_lower = put_center.saturating_sub(wing_width);
        let put_upper = put_center;
        let call_lower = call_center;
        let call_upper = call_center.saturating_add(wing_width);

        let legs = vec![
            // Put spread legs
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: put_lower,
                    maturity,
                    is_call: false,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: put_upper,
                    maturity,
                    is_call: false,
                    is_long: false,
                    size,
                    collateral: size,
                },
                ratio: -1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            // Call spread legs
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: call_lower,
                    maturity,
                    is_call: true,
                    is_long: false,
                    size,
                    collateral: size,
                },
                ratio: -1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: call_upper,
                    maturity,
                    is_call: true,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
        ];

        LeggedPosition {
            legs,
            total_collateral: size * 4,
            strategy_type: StrategyType::IronCondor,
            risk_params: RiskParameters {
                max_loss: size,
                max_leverage: 4.0,
                min_collateral_ratio: 0.25,
                greek_limits: GreekLimits {
                    max_delta: 0.2,
                    max_gamma: 0.05,
                    max_vega: 0.15,
                    max_theta: 0.1,
                },
            },
        }
    }

    pub fn create_calendar_spread(
        &self,
        strike: U256,
        near_maturity: U256,
        far_maturity: U256,
        size: U256,
        is_call: bool,
    ) -> LeggedPosition {
        let legs = vec![
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike,
                    maturity: near_maturity,
                    is_call,
                    is_long: false,
                    size,
                    collateral: size,
                },
                ratio: -1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike,
                    maturity: far_maturity,
                    is_call,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
        ];

        LeggedPosition {
            legs,
            total_collateral: size * 2,
            strategy_type: StrategyType::CalendarSpread,
            risk_params: RiskParameters {
                max_loss: size,
                max_leverage: 2.0,
                min_collateral_ratio: 0.5,
                greek_limits: GreekLimits {
                    max_delta: 0.3,
                    max_gamma: 0.1,
                    max_vega: 0.2,
                    max_theta: 0.1,
                },
            },
        }
    }

    pub fn create_diagonal_spread(
        &self,
        near_strike: U256,
        far_strike: U256,
        near_maturity: U256,
        far_maturity: U256,
        size: U256,
        is_call: bool,
    ) -> LeggedPosition {
        let legs = vec![
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: near_strike,
                    maturity: near_maturity,
                    is_call,
                    is_long: false,
                    size,
                    collateral: size,
                },
                ratio: -1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: far_strike,
                    maturity: far_maturity,
                    is_call,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
        ];

        LeggedPosition {
            legs,
            total_collateral: size * 2,
            strategy_type: StrategyType::DiagonalSpread,
            risk_params: RiskParameters {
                max_loss: size,
                max_leverage: 2.0,
                min_collateral_ratio: 0.5,
                greek_limits: GreekLimits {
                    max_delta: 0.3,
                    max_gamma: 0.1,
                    max_vega: 0.2,
                    max_theta: 0.1,
                },
            },
        }
    }

    pub fn create_ratio_spread(
        &self,
        strike_short: U256,
        strike_long: U256,
        maturity: U256,
        size: U256,
        ratio: i32,
        is_call: bool,
    ) -> LeggedPosition {
        let legs = vec![
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: strike_short,
                    maturity,
                    is_call,
                    is_long: false,
                    size: size * U256::from(ratio as u64),
                    collateral: size * U256::from(ratio as u64),
                },
                ratio: -ratio,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: strike_long,
                    maturity,
                    is_call,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
        ];

        LeggedPosition {
            legs,
            total_collateral: size * (U256::from(ratio as u64) + U256::from(1)),
            strategy_type: StrategyType::RatioSpread,
            risk_params: RiskParameters {
                max_loss: size,
                max_leverage: ratio as f64 + 1.0,
                min_collateral_ratio: 1.0 / (ratio as f64 + 1.0),
                greek_limits: GreekLimits {
                    max_delta: 0.4,
                    max_gamma: 0.15,
                    max_vega: 0.25,
                    max_theta: 0.15,
                },
            },
        }
    }

    pub fn create_straddle_strangle(
        &self,
        call_strike: U256,
        put_strike: U256,
        maturity: U256,
        size: U256,
        is_straddle: bool,
    ) -> LeggedPosition {
        let (actual_call_strike, actual_put_strike) = if is_straddle {
            (call_strike, call_strike)
        } else {
            (call_strike, put_strike)
        };

        let legs = vec![
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: actual_call_strike,
                    maturity,
                    is_call: true,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
            OptionLeg {
                position: OptionPosition {
                    token_id: U256::zero(),
                    strike: actual_put_strike,
                    maturity,
                    is_call: false,
                    is_long: true,
                    size,
                    collateral: size,
                },
                ratio: 1,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            },
        ];

        LeggedPosition {
            legs,
            total_collateral: size * 2,
            strategy_type: StrategyType::StraddleStrangle,
            risk_params: RiskParameters {
                max_loss: size * 2,
                max_leverage: 2.0,
                min_collateral_ratio: 0.5,
                greek_limits: GreekLimits {
                    max_delta: 0.1,
                    max_gamma: 0.2,
                    max_vega: 0.3,
                    max_theta: 0.2,
                },
            },
        }
    }

    pub fn create_custom_strategy(
        &self,
        legs: Vec<(OptionPosition, i32)>,
        strategy_name: String,
        custom_risk_params: Option<RiskParameters>,
    ) -> LeggedPosition {
        let mut total_collateral = U256::zero();
        let mut option_legs = Vec::new();

        for (position, ratio) in legs {
            total_collateral = total_collateral.saturating_add(position.collateral);
            option_legs.push(OptionLeg {
                position,
                ratio,
                delta: 0.0,
                gamma: 0.0,
                theta: 0.0,
                vega: 0.0,
            });
        }

        let default_risk_params = RiskParameters {
            max_loss: total_collateral,
            max_leverage: 3.0,
            min_collateral_ratio: 0.33,
            greek_limits: GreekLimits {
                max_delta: 0.5,
                max_gamma: 0.2,
                max_vega: 0.3,
                max_theta: 0.2,
            },
        };

        LeggedPosition {
            legs: option_legs,
            total_collateral,
            strategy_type: StrategyType::Custom,
            risk_params: custom_risk_params.unwrap_or(default_risk_params),
        }
    }

    pub fn calculate_greeks(&self, position: &mut LeggedPosition) {
        for leg in &mut position.legs {
            let spot_price = self.market_data.spot_price;
            let strike = leg.position.strike;
            let time_to_maturity = self.calculate_time_to_maturity(leg.position.maturity);
            let volatility = self.market_data.volatility;
            let interest_rate = self.market_data.interest_rate;

            let (delta, gamma, theta, vega) = self.black_scholes_greeks(
                spot_price,
                strike,
                time_to_maturity,
                volatility,
                interest_rate,
                leg.position.is_call,
            );

            leg.delta = delta * leg.ratio as f64;
            leg.gamma = gamma * leg.ratio as f64;
            leg.theta = theta * leg.ratio as f64;
            leg.vega = vega * leg.ratio as f64;
        }
    }

    fn calculate_time_to_maturity(&self, maturity: U256) -> f64 {
        let current_time = U256::from(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());
        
        if maturity <= current_time {
            return 0.0;
        }

        let seconds_to_maturity = maturity.saturating_sub(current_time);
        seconds_to_maturity.as_u128() as f64 / (365.0 * 24.0 * 60.0 * 60.0)
    }

    pub fn calculate_implied_volatility(
        &self,
        option_price: f64,
        spot: U256,
        strike: U256,
        time: f64,
        rate: f64,
        is_call: bool,
    ) -> Option<f64> {
        let s = spot.as_u128() as f64;
        let k = strike.as_u128() as f64;
        let r = rate;
        let t = time;

        let mut sigma = 0.5; // Initial guess
        let mut diff = 1.0;
        let tolerance = 1e-5;
        let max_iterations = 100;
        let mut iterations = 0;

        while diff.abs() > tolerance && iterations < max_iterations {
            let (price, vega) = self.black_scholes_price_and_vega(s, k, t, sigma, r, is_call);
            diff = option_price - price;
            
            if vega.abs() < 1e-10 {
                return None;
            }

            sigma += diff / vega;
            if sigma <= 0.0 {
                sigma = 0.0001;
            }

            iterations += 1;
        }

        if iterations == max_iterations {
            None
        } else {
            Some(sigma)
        }
    }

    fn black_scholes_price_and_vega(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        volatility: f64,
        rate: f64,
        is_call: bool,
    ) -> (f64, f64) {
        let d1 = (f64::ln(spot / strike) + (rate + volatility * volatility / 2.0) * time)
            / (volatility * f64::sqrt(time));
        let d2 = d1 - volatility * f64::sqrt(time);

        let n_d1 = self.normal_cdf(d1);
        let n_d2 = self.normal_cdf(d2);
        let n_prime_d1 = self.normal_pdf(d1);

        let price = if is_call {
            spot * n_d1 - strike * f64::exp(-rate * time) * n_d2
        } else {
            strike * f64::exp(-rate * time) * self.normal_cdf(-d2) - spot * self.normal_cdf(-d1)
        };

        let vega = spot * f64::sqrt(time) * n_prime_d1;

        (price, vega)
    }

    pub fn calculate_option_price(
        &self,
        position: &OptionPosition,
        model: PricingModel,
    ) -> f64 {
        let spot = self.market_data.spot_price.as_u128() as f64;
        let strike = position.strike.as_u128() as f64;
        let time = self.calculate_time_to_maturity(position.maturity);
        let vol = self.market_data.volatility;
        let rate = self.market_data.interest_rate;

        match model {
            PricingModel::BlackScholes => {
                let (price, _) = self.black_scholes_price_and_vega(
                    spot,
                    strike,
                    time,
                    vol,
                    rate,
                    position.is_call,
                );
                price
            },
            PricingModel::Binomial => {
                self.binomial_price(spot, strike, time, vol, rate, position.is_call)
            },
            PricingModel::MonteCarlo => {
                self.monte_carlo_price(spot, strike, time, vol, rate, position.is_call)
            },
        }
    }

    fn binomial_price(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        volatility: f64,
        rate: f64,
        is_call: bool,
    ) -> f64 {
        let steps = 100;
        let dt = time / steps as f64;
        let u = f64::exp(volatility * f64::sqrt(dt));
        let d = 1.0 / u;
        let p = (f64::exp(rate * dt) - d) / (u - d);

        let mut prices = vec![0.0; steps + 1];
        for i in 0..=steps {
            let spot_t = spot * u.powi(i as i32) * d.powi((steps - i) as i32);
            prices[i] = if is_call {
                f64::max(0.0, spot_t - strike)
            } else {
                f64::max(0.0, strike - spot_t)
            };
        }

        for step in (0..steps).rev() {
            for i in 0..=step {
                prices[i] = f64::exp(-rate * dt) * (p * prices[i + 1] + (1.0 - p) * prices[i]);
            }
        }

        prices[0]
    }

    fn monte_carlo_price(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        volatility: f64,
        rate: f64,
        is_call: bool,
    ) -> f64 {
        let paths = 10000;
        let mut rng = rand::thread_rng();
        let drift = (rate - 0.5 * volatility * volatility) * time;
        let vol_sqrt = volatility * f64::sqrt(time);
        
        let mut sum = 0.0;
        for _ in 0..paths {
            let z: f64 = rand::distributions::Standard.sample(&mut rng);
            let spot_t = spot * f64::exp(drift + vol_sqrt * z);
            
            let payoff = if is_call {
                f64::max(0.0, spot_t - strike)
            } else {
                f64::max(0.0, strike - spot_t)
            };
            
            sum += payoff;
        }

        f64::exp(-rate * time) * sum / paths as f64
    }

    fn black_scholes_greeks(
        &self,
        spot: U256,
        strike: U256,
        time: f64,
        volatility: f64,
        rate: f64,
        is_call: bool,
    ) -> (f64, f64, f64, f64) {
        let s = spot.as_u128() as f64;
        let k = strike.as_u128() as f64;
        let v = volatility;
        let r = rate;
        let t = time;

        let d1 = (f64::ln(s / k) + (r + v * v / 2.0) * t) / (v * f64::sqrt(t));
        let d2 = d1 - v * f64::sqrt(t);

        let n_d1 = self.normal_cdf(d1);
        let n_d2 = self.normal_cdf(d2);
        let n_prime_d1 = self.normal_pdf(d1);

        let delta = if is_call { n_d1 } else { n_d1 - 1.0 };
        let gamma = n_prime_d1 / (s * v * f64::sqrt(t));
        let theta = -s * n_prime_d1 * v / (2.0 * f64::sqrt(t)) - r * k * f64::exp(-r * t) * n_d2;
        let vega = s * f64::sqrt(t) * n_prime_d1;

        (delta, gamma, theta, vega)
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + libm::erf(x / f64::sqrt(2.0)))
    }

    fn normal_pdf(&self, x: f64) -> f64 {
        f64::exp(-x * x / 2.0) / f64::sqrt(2.0 * std::f64::consts::PI)
    }

    pub fn compose_strategy(&self, strategy_type: AdvancedStrategyType, params: &StrategyParams) -> Result<Strategy, StrategyError> {
        let legs = match strategy_type {
            // Bull Strategy Compositions
            AdvancedStrategyType::BullCallSpread => self.compose_bull_call_spread(params),
            AdvancedStrategyType::BullPutSpread => self.compose_bull_put_spread(params),
            AdvancedStrategyType::InTheMoneyBullCallSpread => self.compose_itm_bull_call_spread(params),
            // ... [Previous bull compositions remain unchanged]
            
            // New Bull Strategy Compositions
            AdvancedStrategyType::BullishDragonSpread => {
                let mut legs = Vec::new();
                legs.extend(self.compose_bull_call_spread(params)?);
                legs.extend(self.compose_bull_put_spread(params)?);
                legs.extend(self.compose_long_butterfly(params)?);
                Ok(legs)
            },
            AdvancedStrategyType::BullishPhoenixSpread => {
                let mut legs = Vec::new();
                legs.extend(self.compose_bull_call_spread(params)?);
                legs.extend(self.compose_bull_put_spread(params)?);
                legs.extend(self.compose_calendar_spread(params)?);
                Ok(legs)
            },
            // ... [Additional bull compositions]

            // Bear Strategy Compositions
            AdvancedStrategyType::BearCallSpread => self.compose_bear_call_spread(params),
            AdvancedStrategyType::BearPutSpread => self.compose_bear_put_spread(params),
            // ... [Previous bear compositions remain unchanged]
            
            // New Bear Strategy Compositions
            AdvancedStrategyType::BearishDragonSpread => {
                let mut legs = Vec::new();
                legs.extend(self.compose_bear_call_spread(params)?);
                legs.extend(self.compose_bear_put_spread(params)?);
                legs.extend(self.compose_short_butterfly(params)?);
                Ok(legs)
            },
            AdvancedStrategyType::BearishPhoenixSpread => {
                let mut legs = Vec::new();
                legs.extend(self.compose_bear_call_spread(params)?);
                legs.extend(self.compose_bear_put_spread(params)?);
                legs.extend(self.compose_reverse_calendar_spread(params)?);
                Ok(legs)
            },
            // ... [Additional bear compositions]

            // Neutral Strategy Compositions
            AdvancedStrategyType::LongIronCondor => self.compose_long_iron_condor(params),
            AdvancedStrategyType::ShortIronCondor => self.compose_short_iron_condor(params),
            // ... [Previous neutral compositions remain unchanged]
            
            // New Neutral Strategy Compositions
            AdvancedStrategyType::NeutralDragonSpread => {
                let mut legs = Vec::new();
                legs.extend(self.compose_iron_condor(params)?);
                legs.extend(self.compose_butterfly(params)?);
                legs.extend(self.compose_straddle(params)?);
                Ok(legs)
            },
            AdvancedStrategyType::NeutralPhoenixSpread => {
                let mut legs = Vec::new();
                legs.extend(self.compose_iron_butterfly(params)?);
                legs.extend(self.compose_calendar_spread(params)?);
                legs.extend(self.compose_ratio_spread(params)?);
                Ok(legs)
            },
            // ... [Additional neutral compositions]
        };

        Ok(Strategy {
            strategy_type,
            legs: legs?,
            params: params.clone(),
            risk_metrics: self.calculate_risk_metrics(&legs?, params)?,
        })
    }

    fn calculate_risk_metrics(&self, legs: &[StrategyLeg], params: &StrategyParams) -> Result<RiskMetrics, StrategyError> {
        let mut metrics = RiskMetrics::default();
        
        // Calculate Greeks
        for leg in legs {
            metrics.delta += self.calculate_leg_delta(leg, params)?;
            metrics.gamma += self.calculate_leg_gamma(leg, params)?;
            metrics.theta += self.calculate_leg_theta(leg, params)?;
            metrics.vega += self.calculate_leg_vega(leg, params)?;
            metrics.rho += self.calculate_leg_rho(leg, params)?;
        }
        
        // Calculate position sizing using Kelly Criterion
        metrics.position_size = self.calculate_kelly_position_size(legs, params)?;
        
        // Calculate risk-adjusted returns
        metrics.sharpe_ratio = self.calculate_sharpe_ratio(legs, params)?;
        metrics.sortino_ratio = self.calculate_sortino_ratio(legs, params)?;
        metrics.calmar_ratio = self.calculate_calmar_ratio(legs, params)?;
        
        // Calculate risk measures
        metrics.var_95 = self.calculate_value_at_risk(legs, params, 0.95)?;
        metrics.cvar_95 = self.calculate_conditional_var(legs, params, 0.95)?;
        metrics.max_drawdown = self.calculate_max_drawdown(legs, params)?;
        
        Ok(metrics)
    }

    fn calculate_kelly_position_size(&self, legs: &[StrategyLeg], params: &StrategyParams) -> Result<f64, StrategyError> {
        let win_prob = self.optimizer.calculate_win_probability(legs, params)?;
        let avg_win = self.optimizer.calculate_average_win(legs, params)?;
        let avg_loss = self.optimizer.calculate_average_loss(legs, params)?;
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        Ok(kelly_fraction.max(0.0).min(1.0))
    }

    fn calculate_sharpe_ratio(&self, legs: &[StrategyLeg], params: &StrategyParams) -> Result<f64, StrategyError> {
        let returns = self.calculate_historical_returns(legs, params)?;
        let excess_returns: Vec<f64> = returns.iter()
            .map(|r| r - params.risk_free_rate)
            .collect();
            
        let mean_excess_return = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let std_dev = self.calculate_standard_deviation(&excess_returns)?;
        
        Ok(mean_excess_return / std_dev)
    }

    fn calculate_sortino_ratio(&self, legs: &[StrategyLeg], params: &StrategyParams) -> Result<f64, StrategyError> {
        let returns = self.calculate_historical_returns(legs, params)?;
        let excess_returns: Vec<f64> = returns.iter()
            .map(|r| r - params.risk_free_rate)
            .collect();
            
        let mean_excess_return = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let downside_returns: Vec<f64> = excess_returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
            
        let downside_std = self.calculate_standard_deviation(&downside_returns)?;
        
        Ok(mean_excess_return / downside_std)
    }

    fn calculate_calmar_ratio(&self, legs: &[StrategyLeg], params: &StrategyParams) -> Result<f64, StrategyError> {
        let returns = self.calculate_historical_returns(legs, params)?;
        let annual_return = returns.iter().sum::<f64>();
        let max_drawdown = self.calculate_max_drawdown(legs, params)?;
        
        Ok(annual_return / max_drawdown)
    }

    fn calculate_value_at_risk(&self, legs: &[StrategyLeg], params: &StrategyParams, confidence: f64) -> Result<f64, StrategyError> {
        let returns = self.calculate_historical_returns(legs, params)?;
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * returns.len() as f64).floor() as usize;
        Ok(-sorted_returns[index])
    }

    fn calculate_conditional_var(&self, legs: &[StrategyLeg], params: &StrategyParams, confidence: f64) -> Result<f64, StrategyError> {
        let returns = self.calculate_historical_returns(legs, params)?;
        let var = self.calculate_value_at_risk(legs, params, confidence)?;
        
        let tail_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r <= -var)
            .cloned()
            .collect();
            
        Ok(-tail_returns.iter().sum::<f64>() / tail_returns.len() as f64)
    }

    fn calculate_max_drawdown(&self, legs: &[StrategyLeg], params: &StrategyParams) -> Result<f64, StrategyError> {
        let returns = self.calculate_historical_returns(legs, params)?;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;
        let mut current_value = 1.0;
        
        for &ret in &returns {
            current_value *= 1.0 + ret;
            peak = peak.max(current_value);
            let drawdown = (peak - current_value) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }
        
        Ok(max_drawdown)
    }

    fn calculate_standard_deviation(&self, values: &[f64]) -> Result<f64, StrategyError> {
        if values.is_empty() {
            return Err(StrategyError::InsufficientData);
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
            
        Ok(variance.sqrt())
    }
}
