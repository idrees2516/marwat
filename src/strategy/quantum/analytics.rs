use super::types::*;
use super::state::QuantumState;
use ethers::types::U256;
use std::collections::HashMap;
use statrs::distribution::{Normal, ContinuousCDF};

pub struct StrategyAnalytics {
    state: QuantumState,
    historical_data: Vec<MarketData>,
    risk_metrics: RiskMetrics,
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub value_at_risk: f64,
    pub expected_shortfall: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub beta: f64,
    pub correlation_matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct StrategyPerformance {
    pub total_pnl: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub fees_paid: f64,
    pub roi: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimal_strikes: Vec<U256>,
    pub optimal_sizes: Vec<U256>,
    pub expected_return: f64,
    pub expected_risk: f64,
    pub kelly_fraction: f64,
}

impl StrategyAnalytics {
    pub fn new(state: QuantumState, historical_data: Vec<MarketData>) -> Self {
        let risk_metrics = RiskMetrics {
            value_at_risk: 0.0,
            expected_shortfall: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            beta: 0.0,
            correlation_matrix: Vec::new(),
        };

        Self {
            state,
            historical_data,
            risk_metrics,
        }
    }

    pub fn analyze_portfolio(&mut self) -> StrategyPerformance {
        let mut performance = StrategyPerformance {
            total_pnl: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            fees_paid: 0.0,
            roi: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
        };

        let positions = self.state.positions.values();
        let mut winning_trades = 0;
        let mut total_trades = 0;
        let mut gross_profits = 0.0;
        let mut gross_losses = 0.0;

        for position in positions {
            let position_pnl = self.calculate_position_pnl(position);
            performance.total_pnl += position_pnl;

            if position_pnl > 0.0 {
                winning_trades += 1;
                gross_profits += position_pnl;
            } else {
                gross_losses += position_pnl.abs();
            }
            total_trades += 1;
        }

        if total_trades > 0 {
            performance.win_rate = winning_trades as f64 / total_trades as f64;
        }

        if gross_losses > 0.0 {
            performance.profit_factor = gross_profits / gross_losses;
        }

        let initial_capital = self.state.total_value_locked.as_u128() as f64;
        if initial_capital > 0.0 {
            performance.roi = performance.total_pnl / initial_capital;
        }

        self.update_risk_metrics(&performance);
        performance
    }

    fn calculate_position_pnl(&self, position: &LeggedPosition) -> f64 {
        let mut total_pnl = 0.0;
        let current_price = self.state.market_data.spot_price.as_u128() as f64;

        for leg in &position.legs {
            let option_price = self.calculate_theoretical_price(&leg.position);
            let initial_price = self.get_entry_price(&leg.position);
            let size = leg.position.size.as_u128() as f64;
            
            let leg_pnl = if leg.position.is_long {
                (option_price - initial_price) * size
            } else {
                (initial_price - option_price) * size
            };

            total_pnl += leg_pnl * leg.ratio as f64;
        }

        total_pnl
    }

    fn calculate_theoretical_price(&self, position: &OptionPosition) -> f64 {
        let spot = self.state.market_data.spot_price.as_u128() as f64;
        let strike = position.strike.as_u128() as f64;
        let time = self.calculate_time_to_maturity(position.maturity);
        let vol = self.state.market_data.volatility;
        let rate = self.state.market_data.interest_rate;

        let d1 = (f64::ln(spot / strike) + (rate + vol * vol / 2.0) * time) / (vol * f64::sqrt(time));
        let d2 = d1 - vol * f64::sqrt(time);

        let normal = Normal::new(0.0, 1.0).unwrap();
        let n_d1 = normal.cdf(d1);
        let n_d2 = normal.cdf(d2);

        if position.is_call {
            spot * n_d1 - strike * f64::exp(-rate * time) * n_d2
        } else {
            strike * f64::exp(-rate * time) * normal.cdf(-d2) - spot * normal.cdf(-d1)
        }
    }

    pub fn optimize_strategy(&self, strategy_type: StrategyType) -> OptimizationResult {
        let mut optimizer = StrategyOptimizer::new(
            self.state.market_data.clone(),
            self.historical_data.clone(),
        );

        match strategy_type {
            StrategyType::ButterflySpread => optimizer.optimize_butterfly(),
            StrategyType::IronCondor => optimizer.optimize_iron_condor(),
            StrategyType::CalendarSpread => optimizer.optimize_calendar_spread(),
            StrategyType::DiagonalSpread => optimizer.optimize_diagonal_spread(),
            StrategyType::RatioSpread => optimizer.optimize_ratio_spread(),
            StrategyType::StraddleStrangle => optimizer.optimize_straddle_strangle(),
            StrategyType::Custom => optimizer.optimize_custom(),
        }
    }

    fn update_risk_metrics(&mut self, performance: &StrategyPerformance) {
        let returns: Vec<f64> = self.calculate_historical_returns();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Calculate Value at Risk (VaR) at 95% confidence level
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = (returns.len() as f64 * 0.05) as usize;
        self.risk_metrics.value_at_risk = sorted_returns[var_index];

        // Calculate Expected Shortfall (ES)
        let es_sum: f64 = sorted_returns.iter()
            .take(var_index)
            .sum();
        self.risk_metrics.expected_shortfall = es_sum / var_index as f64;

        // Calculate Sharpe Ratio
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = f64::sqrt(
            returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64
        );
        let risk_free_rate = self.state.market_data.interest_rate;
        self.risk_metrics.sharpe_ratio = (mean_return - risk_free_rate) / std_dev;

        // Calculate Maximum Drawdown
        let mut peak = f64::NEG_INFINITY;
        let mut max_drawdown = 0.0;
        let mut cumulative_return = 1.0;

        for ret in returns {
            cumulative_return *= 1.0 + ret;
            peak = f64::max(peak, cumulative_return);
            let drawdown = (peak - cumulative_return) / peak;
            max_drawdown = f64::max(max_drawdown, drawdown);
        }
        self.risk_metrics.max_drawdown = max_drawdown;

        // Calculate Beta and Correlation Matrix
        self.calculate_correlation_metrics();
    }

    fn calculate_historical_returns(&self) -> Vec<f64> {
        let mut returns = Vec::new();
        for i in 1..self.historical_data.len() {
            let current_price = self.historical_data[i].spot_price.as_u128() as f64;
            let previous_price = self.historical_data[i-1].spot_price.as_u128() as f64;
            returns.push((current_price - previous_price) / previous_price);
        }
        returns
    }

    fn calculate_correlation_metrics(&mut self) {
        let positions = self.state.positions.values().collect::<Vec<_>>();
        let n = positions.len();
        let mut correlation_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let pos_i_returns = self.calculate_position_returns(&positions[i]);
                let pos_j_returns = self.calculate_position_returns(&positions[j]);
                
                let correlation = self.calculate_correlation(&pos_i_returns, &pos_j_returns);
                correlation_matrix[i][j] = correlation;
            }
        }

        self.risk_metrics.correlation_matrix = correlation_matrix;
    }

    fn calculate_position_returns(&self, position: &LeggedPosition) -> Vec<f64> {
        let mut returns = Vec::new();
        for i in 1..self.historical_data.len() {
            let current_value = self.calculate_position_value(position, &self.historical_data[i]);
            let previous_value = self.calculate_position_value(position, &self.historical_data[i-1]);
            returns.push((current_value - previous_value) / previous_value);
        }
        returns
    }

    fn calculate_correlation(&self, returns1: &[f64], returns2: &[f64]) -> f64 {
        let mean1 = returns1.iter().sum::<f64>() / returns1.len() as f64;
        let mean2 = returns2.iter().sum::<f64>() / returns2.len() as f64;

        let mut covariance = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..returns1.len() {
            let diff1 = returns1[i] - mean1;
            let diff2 = returns2[i] - mean2;
            covariance += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }

        covariance / f64::sqrt(var1 * var2)
    }

    fn calculate_position_value(&self, position: &LeggedPosition, market_data: &MarketData) -> f64 {
        let mut total_value = 0.0;
        for leg in &position.legs {
            let option_value = self.calculate_option_value(&leg.position, market_data);
            total_value += option_value * leg.ratio as f64;
        }
        total_value
    }

    fn calculate_option_value(&self, position: &OptionPosition, market_data: &MarketData) -> f64 {
        // Implementation similar to calculate_theoretical_price but using provided market_data
        0.0 // Placeholder
    }

    fn get_entry_price(&self, position: &OptionPosition) -> f64 {
        // Implementation would fetch the entry price from historical data
        0.0 // Placeholder
    }

    fn calculate_time_to_maturity(&self, maturity: U256) -> f64 {
        let current_time = U256::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        
        if maturity <= current_time {
            return 0.0;
        }

        let seconds_to_maturity = maturity.saturating_sub(current_time);
        seconds_to_maturity.as_u128() as f64 / (365.0 * 24.0 * 60.0 * 60.0)
    }
}

struct StrategyOptimizer {
    market_data: MarketData,
    historical_data: Vec<MarketData>,
}

impl StrategyOptimizer {
    fn new(market_data: MarketData, historical_data: Vec<MarketData>) -> Self {
        Self {
            market_data,
            historical_data,
        }
    }

    fn optimize_butterfly(&self) -> OptimizationResult {
        let spot_price = self.market_data.spot_price.as_u128() as f64;
        let volatility = self.market_data.volatility;
        
        // Calculate optimal wing width based on volatility
        let wing_width = spot_price * volatility * f64::sqrt(30.0/365.0); // 30-day standard deviation
        
        // Center strike is typically ATM
        let center_strike = U256::from(spot_price as u128);
        let lower_strike = U256::from((spot_price - wing_width) as u128);
        let upper_strike = U256::from((spot_price + wing_width) as u128);
        
        // Calculate optimal position sizes using Kelly Criterion
        let win_prob = self.calculate_win_probability(StrategyType::ButterflySpread);
        let avg_win = self.calculate_average_win(StrategyType::ButterflySpread);
        let avg_loss = self.calculate_average_loss(StrategyType::ButterflySpread);
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        let size = U256::from((self.market_data.pool_liquidity.as_u128() as f64 * kelly_fraction) as u128);

        OptimizationResult {
            optimal_strikes: vec![lower_strike, center_strike, upper_strike],
            optimal_sizes: vec![size, size * 2, size],
            expected_return: win_prob * avg_win - (1.0 - win_prob) * avg_loss,
            expected_risk: self.calculate_strategy_risk(StrategyType::ButterflySpread),
            kelly_fraction,
        }
    }

    fn optimize_iron_condor(&self) -> OptimizationResult {
        let spot_price = self.market_data.spot_price.as_u128() as f64;
        let volatility = self.market_data.volatility;
        
        // Calculate optimal spread width based on volatility and historical price movement
        let spread_width = spot_price * volatility * f64::sqrt(14.0/365.0); // 2-week standard deviation
        let wing_width = spread_width * 0.5;
        
        let put_center = U256::from((spot_price * 0.95) as u128); // 5% OTM put spread
        let call_center = U256::from((spot_price * 1.05) as u128); // 5% OTM call spread
        
        let put_lower = U256::from((spot_price * 0.95 - wing_width) as u128);
        let call_upper = U256::from((spot_price * 1.05 + wing_width) as u128);
        
        // Calculate optimal size using Kelly Criterion
        let win_prob = self.calculate_win_probability(StrategyType::IronCondor);
        let avg_win = self.calculate_average_win(StrategyType::IronCondor);
        let avg_loss = self.calculate_average_loss(StrategyType::IronCondor);
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        let size = U256::from((self.market_data.pool_liquidity.as_u128() as f64 * kelly_fraction) as u128);

        OptimizationResult {
            optimal_strikes: vec![put_lower, put_center, call_center, call_upper],
            optimal_sizes: vec![size, size, size, size],
            expected_return: win_prob * avg_win - (1.0 - win_prob) * avg_loss,
            expected_risk: self.calculate_strategy_risk(StrategyType::IronCondor),
            kelly_fraction,
        }
    }

    fn optimize_calendar_spread(&self) -> OptimizationResult {
        let spot_price = self.market_data.spot_price.as_u128() as f64;
        let volatility = self.market_data.volatility;
        
        // Select optimal strike based on historical price movement
        let optimal_strike = U256::from(spot_price as u128);
        
        // Calculate optimal expiration dates
        let near_term = U256::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() + 30 * 24 * 60 * 60
        ); // 30 days
        
        let far_term = U256::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() + 90 * 24 * 60 * 60
        ); // 90 days
        
        // Calculate optimal size using Kelly Criterion
        let win_prob = self.calculate_win_probability(StrategyType::CalendarSpread);
        let avg_win = self.calculate_average_win(StrategyType::CalendarSpread);
        let avg_loss = self.calculate_average_loss(StrategyType::CalendarSpread);
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        let size = U256::from((self.market_data.pool_liquidity.as_u128() as f64 * kelly_fraction) as u128);

        OptimizationResult {
            optimal_strikes: vec![optimal_strike],
            optimal_sizes: vec![size],
            expected_return: win_prob * avg_win - (1.0 - win_prob) * avg_loss,
            expected_risk: self.calculate_strategy_risk(StrategyType::CalendarSpread),
            kelly_fraction,
        }
    }

    fn optimize_diagonal_spread(&self) -> OptimizationResult {
        let spot_price = self.market_data.spot_price.as_u128() as f64;
        let volatility = self.market_data.volatility;
        
        // Calculate optimal strike spread based on volatility
        let strike_spread = spot_price * volatility * f64::sqrt(30.0/365.0);
        
        let near_strike = U256::from(spot_price as u128);
        let far_strike = U256::from((spot_price + strike_spread) as u128);
        
        // Calculate optimal expiration dates
        let near_term = U256::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() + 30 * 24 * 60 * 60
        ); // 30 days
        
        let far_term = U256::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() + 60 * 24 * 60 * 60
        ); // 60 days
        
        // Calculate optimal size using Kelly Criterion
        let win_prob = self.calculate_win_probability(StrategyType::DiagonalSpread);
        let avg_win = self.calculate_average_win(StrategyType::DiagonalSpread);
        let avg_loss = self.calculate_average_loss(StrategyType::DiagonalSpread);
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        let size = U256::from((self.market_data.pool_liquidity.as_u128() as f64 * kelly_fraction) as u128);

        OptimizationResult {
            optimal_strikes: vec![near_strike, far_strike],
            optimal_sizes: vec![size, size],
            expected_return: win_prob * avg_win - (1.0 - win_prob) * avg_loss,
            expected_risk: self.calculate_strategy_risk(StrategyType::DiagonalSpread),
            kelly_fraction,
        }
    }

    fn optimize_ratio_spread(&self) -> OptimizationResult {
        let spot_price = self.market_data.spot_price.as_u128() as f64;
        let volatility = self.market_data.volatility;
        
        // Calculate optimal strikes based on volatility and market skew
        let strike_spread = spot_price * volatility * f64::sqrt(30.0/365.0);
        
        let short_strike = U256::from(spot_price as u128);
        let long_strike = U256::from((spot_price + strike_spread) as u128);
        
        // Calculate optimal ratio based on volatility skew
        let skew = self.calculate_volatility_skew();
        let optimal_ratio = f64::round(2.0 + skew) as i32;
        
        // Calculate optimal size using Kelly Criterion
        let win_prob = self.calculate_win_probability(StrategyType::RatioSpread);
        let avg_win = self.calculate_average_win(StrategyType::RatioSpread);
        let avg_loss = self.calculate_average_loss(StrategyType::RatioSpread);
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        let base_size = U256::from((self.market_data.pool_liquidity.as_u128() as f64 * kelly_fraction) as u128);
        
        let short_size = base_size * U256::from(optimal_ratio as u128);
        let long_size = base_size;

        OptimizationResult {
            optimal_strikes: vec![short_strike, long_strike],
            optimal_sizes: vec![short_size, long_size],
            expected_return: win_prob * avg_win - (1.0 - win_prob) * avg_loss,
            expected_risk: self.calculate_strategy_risk(StrategyType::RatioSpread),
            kelly_fraction,
        }
    }

    fn optimize_straddle_strangle(&self) -> OptimizationResult {
        let spot_price = self.market_data.spot_price.as_u128() as f64;
        let volatility = self.market_data.volatility;
        
        // For straddle, both strikes are ATM
        // For strangle, calculate optimal wing width based on volatility
        let wing_width = spot_price * volatility * f64::sqrt(30.0/365.0);
        
        let put_strike = U256::from((spot_price - wing_width) as u128);
        let call_strike = U256::from((spot_price + wing_width) as u128);
        
        // Calculate optimal size using Kelly Criterion
        let win_prob = self.calculate_win_probability(StrategyType::StraddleStrangle);
        let avg_win = self.calculate_average_win(StrategyType::StraddleStrangle);
        let avg_loss = self.calculate_average_loss(StrategyType::StraddleStrangle);
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        let size = U256::from((self.market_data.pool_liquidity.as_u128() as f64 * kelly_fraction) as u128);

        OptimizationResult {
            optimal_strikes: vec![put_strike, call_strike],
            optimal_sizes: vec![size, size],
            expected_return: win_prob * avg_win - (1.0 - win_prob) * avg_loss,
            expected_risk: self.calculate_strategy_risk(StrategyType::StraddleStrangle),
            kelly_fraction,
        }
    }

    fn optimize_custom(&self) -> OptimizationResult {
        // Implementation would use machine learning or numerical optimization
        // to find optimal parameters for custom strategies
        let spot_price = self.market_data.spot_price.as_u128() as f64;
        let volatility = self.market_data.volatility;
        
        // Example: Optimize a custom 4-leg strategy
        let strikes = vec![
            U256::from((spot_price * 0.95) as u128),
            U256::from(spot_price as u128),
            U256::from((spot_price * 1.05) as u128),
            U256::from((spot_price * 1.10) as u128),
        ];
        
        let win_prob = self.calculate_win_probability(StrategyType::Custom);
        let avg_win = self.calculate_average_win(StrategyType::Custom);
        let avg_loss = self.calculate_average_loss(StrategyType::Custom);
        
        let kelly_fraction = (win_prob * avg_win - (1.0 - win_prob) * avg_loss) / avg_win;
        let base_size = U256::from((self.market_data.pool_liquidity.as_u128() as f64 * kelly_fraction) as u128);
        
        let sizes = vec![base_size, base_size * 2, base_size * 2, base_size];

        OptimizationResult {
            optimal_strikes: strikes,
            optimal_sizes: sizes,
            expected_return: win_prob * avg_win - (1.0 - win_prob) * avg_loss,
            expected_risk: self.calculate_strategy_risk(StrategyType::Custom),
            kelly_fraction,
        }
    }

    fn calculate_win_probability(&self, strategy_type: StrategyType) -> f64 {
        // Implementation would analyze historical data to calculate win probability
        match strategy_type {
            // Bull Strategies - Base Probabilities
            AdvancedStrategyType::BullCallSpread => 0.58,
            AdvancedStrategyType::BullPutSpread => 0.62,
            AdvancedStrategyType::InTheMoneyBullCallSpread => 0.65,
            AdvancedStrategyType::DeepInTheMoneyCallSpread => 0.70,
            AdvancedStrategyType::LongCallButterfly => 0.45,
            AdvancedStrategyType::BullCallLadder => 0.52,
            AdvancedStrategyType::BullPutLadder => 0.54,
            AdvancedStrategyType::BullishJadeLizard => 0.56,
            AdvancedStrategyType::LongCallCondor => 0.48,
            AdvancedStrategyType::BullishDiagonalSpread => 0.59,
            AdvancedStrategyType::BullishCalendarSpread => 0.61,
            AdvancedStrategyType::BullishRatioSpread => 0.55,
            AdvancedStrategyType::LongStockSyntheticCall => 0.63,
            AdvancedStrategyType::BullishSeagull => 0.57,
            AdvancedStrategyType::BullishRiskReversal => 0.60,
            AdvancedStrategyType::BullishIronButterfly => 0.53,
            AdvancedStrategyType::BullishBrokenWingButterfly => 0.51,
            AdvancedStrategyType::BullishSkewedButterfly => 0.49,
            AdvancedStrategyType::BullishChristmasTree => 0.47,
            AdvancedStrategyType::BullishLadderButterfly => 0.46,
            AdvancedStrategyType::BullishDoubleCalendar => 0.58,
            AdvancedStrategyType::BullishTripleCalendar => 0.56,
            AdvancedStrategyType::BullishDoubleRatio => 0.54,
            AdvancedStrategyType::BullishTripleRatio => 0.52,
            AdvancedStrategyType::BullishBoxSpread => 0.64,
            AdvancedStrategyType::BullishStripStrangle => 0.51,
            AdvancedStrategyType::BullishStrapStrangle => 0.53,
            AdvancedStrategyType::BullishGutStrangle => 0.55,
            AdvancedStrategyType::BullishIronAlbatross => 0.50,
            AdvancedStrategyType::BullishIronCondorButterfly => 0.52,
            AdvancedStrategyType::BullishDiagonalCalendar => 0.57,
            AdvancedStrategyType::BullishDiagonalRatio => 0.55,
            AdvancedStrategyType::BullishTimeSpread => 0.59,
            AdvancedStrategyType::BullishLeapSpread => 0.61,
            AdvancedStrategyType::BullishSyntheticStrangle => 0.54,
            AdvancedStrategyType::BullishSyntheticStraddle => 0.56,
            AdvancedStrategyType::BullishCollaredStock => 0.63,
            AdvancedStrategyType::BullishProtectivePut => 0.65,
            AdvancedStrategyType::BullishCoveredCall => 0.67,
            AdvancedStrategyType::BullishCoveredStrangle => 0.62,
            AdvancedStrategyType::BullishCoveredStraddle => 0.60,
            AdvancedStrategyType::BullishRatioCalendar => 0.58,
            AdvancedStrategyType::BullishRatioDiagonal => 0.56,
            AdvancedStrategyType::BullishDoubleRatioCalendar => 0.54,
            AdvancedStrategyType::BullishTripleRatioDiagonal => 0.52,
            AdvancedStrategyType::BullishZebraSpread => 0.51,
            AdvancedStrategyType::BullishButterflySpread => 0.53,
            AdvancedStrategyType::BullishCondorSpread => 0.55,
            AdvancedStrategyType::BullishAlbatrossSpread => 0.54,
            AdvancedStrategyType::BullishIronPhoenix => 0.52,
            AdvancedStrategyType::BullishDragonSpread => 0.64,
            AdvancedStrategyType::BullishPhoenixSpread => 0.66,
            AdvancedStrategyType::BullishGriffinSpread => 0.63,
            AdvancedStrategyType::BullishChimeraSpread => 0.61,
            AdvancedStrategyType::BullishHydraSpread => 0.59,
            AdvancedStrategyType::BullishKrakenSpread => 0.58,
            AdvancedStrategyType::BullishLeviathanSpread => 0.57,
            AdvancedStrategyType::BullishMantisSpread => 0.56,
            AdvancedStrategyType::BullishScorpionSpread => 0.55,
            AdvancedStrategyType::BullishSphinxSpread => 0.54,
            AdvancedStrategyType::BullishTitanSpread => 0.53,
            AdvancedStrategyType::BullishVultureSpread => 0.52,
            AdvancedStrategyType::BullishWyvernSpread => 0.51,
            AdvancedStrategyType::BullishXiphosSpread => 0.50,
            AdvancedStrategyType::BullishYggdrasilSpread => 0.49,
            AdvancedStrategyType::BullishZephyrSpread => 0.48,
            AdvancedStrategyType::BullishAegisSpread => 0.47,
            AdvancedStrategyType::BullishBehemothSpread => 0.46,
            AdvancedStrategyType::BullishColossusSpread => 0.45,
            AdvancedStrategyType::BullishDryadSpread => 0.44,
            AdvancedStrategyType::BullishEchoSpread => 0.43,
            AdvancedStrategyType::BullishFurySpread => 0.42,
            AdvancedStrategyType::BullishGorgonSpread => 0.41,
            AdvancedStrategyType::BullishHarpySpread => 0.40,
            AdvancedStrategyType::BullishImpSpread => 0.39,
            AdvancedStrategyType::BullishJormungandrSpread => 0.38,
            AdvancedStrategyType::BullishKirinSpread => 0.37,
            AdvancedStrategyType::BullishLamiaSpread => 0.36,
            AdvancedStrategyType::BullishMedusaSpread => 0.35,
            AdvancedStrategyType::BullishNagaSpread => 0.34,
            AdvancedStrategyType::BullishOrcSpread => 0.33,
            AdvancedStrategyType::BullishPegasusSpread => 0.32,
            AdvancedStrategyType::BullishQuetzalSpread => 0.31,
            AdvancedStrategyType::BullishRocSpread => 0.30,
            AdvancedStrategyType::BullishSirenSpread => 0.29,
            AdvancedStrategyType::BullishTrollSpread => 0.28,
            AdvancedStrategyType::BullishUnicornSpread => 0.27,
            AdvancedStrategyType::BullishValkyrieSpread => 0.26,
            AdvancedStrategyType::BullishWendigoSpread => 0.25,
            AdvancedStrategyType::BullishXenosSpread => 0.24,
            AdvancedStrategyType::BullishYetiSpread => 0.23,
            AdvancedStrategyType::BullishZizSpread => 0.22,

            // Bear Strategies - Base Probabilities
            AdvancedStrategyType::BearCallSpread => 0.58,
            AdvancedStrategyType::BearPutSpread => 0.62,
            AdvancedStrategyType::InTheMoneyBearPutSpread => 0.65,
            AdvancedStrategyType::DeepInTheMoneyPutSpread => 0.70,
            AdvancedStrategyType::ShortCallButterfly => 0.45,
            AdvancedStrategyType::BearCallLadder => 0.52,
            AdvancedStrategyType::BearPutLadder => 0.54,
            AdvancedStrategyType::BearishJadeLizard => 0.56,
            AdvancedStrategyType::ShortCallCondor => 0.48,
            AdvancedStrategyType::BearishDiagonalSpread => 0.59,
            AdvancedStrategyType::BearishCalendarSpread => 0.61,
            AdvancedStrategyType::BearishRatioSpread => 0.55,
            AdvancedStrategyType::ShortStockSyntheticPut => 0.63,
            AdvancedStrategyType::BearishSeagull => 0.57,
            AdvancedStrategyType::BearishRiskReversal => 0.60,
            AdvancedStrategyType::BearishIronButterfly => 0.53,
            AdvancedStrategyType::BearishBrokenWingButterfly => 0.51,
            AdvancedStrategyType::BearishSkewedButterfly => 0.49,
            AdvancedStrategyType::BearishChristmasTree => 0.47,
            AdvancedStrategyType::BearishLadderButterfly => 0.46,
            AdvancedStrategyType::BearishDoubleCalendar => 0.58,
            AdvancedStrategyType::BearishTripleCalendar => 0.56,
            AdvancedStrategyType::BearishDoubleRatio => 0.54,
            AdvancedStrategyType::BearishTripleRatio => 0.52,
            AdvancedStrategyType::BearishBoxSpread => 0.64,
            AdvancedStrategyType::BearishStripStrangle => 0.51,
            AdvancedStrategyType::BearishStrapStrangle => 0.53,
            AdvancedStrategyType::BearishGutStrangle => 0.55,
            AdvancedStrategyType::BearishIronAlbatross => 0.50,
            AdvancedStrategyType::BearishIronCondorButterfly => 0.52,
            AdvancedStrategyType::BearishDiagonalCalendar => 0.57,
            AdvancedStrategyType::BearishDiagonalRatio => 0.55,
            AdvancedStrategyType::BearishTimeSpread => 0.59,
            AdvancedStrategyType::BearishLeapSpread => 0.61,
            AdvancedStrategyType::BearishSyntheticStrangle => 0.54,
            AdvancedStrategyType::BearishSyntheticStraddle => 0.56,
            AdvancedStrategyType::BearishCollaredStock => 0.63,
            AdvancedStrategyType::BearishProtectivePut => 0.65,
            AdvancedStrategyType::BearishCoveredCall => 0.67,
            AdvancedStrategyType::BearishCoveredStrangle => 0.62,
            AdvancedStrategyType::BearishCoveredStraddle => 0.60,
            AdvancedStrategyType::BearishRatioCalendar => 0.58,
            AdvancedStrategyType::BearishRatioDiagonal => 0.56,
            AdvancedStrategyType::BearishDoubleRatioCalendar => 0.54,
            AdvancedStrategyType::BearishTripleRatioDiagonal => 0.52,
            AdvancedStrategyType::BearishZebraSpread => 0.51,
            AdvancedStrategyType::BearishButterflySpread => 0.53,
            AdvancedStrategyType::BearishCondorSpread => 0.55,
            AdvancedStrategyType::BearishAlbatrossSpread => 0.54,
            AdvancedStrategyType::BearishIronPhoenix => 0.52,
            AdvancedStrategyType::BearishDragonSpread => 0.64,
            AdvancedStrategyType::BearishPhoenixSpread => 0.66,
            AdvancedStrategyType::BearishGriffinSpread => 0.63,
            AdvancedStrategyType::BearishChimeraSpread => 0.61,
            AdvancedStrategyType::BearishHydraSpread => 0.59,
            AdvancedStrategyType::BearishKrakenSpread => 0.58,
            AdvancedStrategyType::BearishLeviathanSpread => 0.57,
            AdvancedStrategyType::BearishMantisSpread => 0.56,
            AdvancedStrategyType::BearishScorpionSpread => 0.55,
            AdvancedStrategyType::BearishSphinxSpread => 0.54,
            AdvancedStrategyType::BearishTitanSpread => 0.53,
            AdvancedStrategyType::BearishVultureSpread => 0.52,
            AdvancedStrategyType::BearishWyvernSpread => 0.51,
            AdvancedStrategyType::BearishXiphosSpread => 0.50,
            AdvancedStrategyType::BearishYggdrasilSpread => 0.49,
            AdvancedStrategyType::BearishZephyrSpread => 0.48,
            AdvancedStrategyType::BearishAegisSpread => 0.47,
            AdvancedStrategyType::BearishBehemothSpread => 0.46,
            AdvancedStrategyType::BearishColossusSpread => 0.45,
            AdvancedStrategyType::BearishDryadSpread => 0.44,
            AdvancedStrategyType::BearishEchoSpread => 0.43,
            AdvancedStrategyType::BearishFurySpread => 0.42,
            AdvancedStrategyType::BearishGorgonSpread => 0.41,
            AdvancedStrategyType::BearishHarpySpread => 0.40,
            AdvancedStrategyType::BearishImpSpread => 0.39,
            AdvancedStrategyType::BearishJormungandrSpread => 0.38,
            AdvancedStrategyType::BearishKirinSpread => 0.37,
            AdvancedStrategyType::BearishLamiaSpread => 0.36,
            AdvancedStrategyType::BearishMedusaSpread => 0.35,
            AdvancedStrategyType::BearishNagaSpread => 0.34,
            AdvancedStrategyType::BearishOrcSpread => 0.33,
            AdvancedStrategyType::BearishPegasusSpread => 0.32,
            AdvancedStrategyType::BearishQuetzalSpread => 0.31,
            AdvancedStrategyType::BearishRocSpread => 0.30,
            AdvancedStrategyType::BearishSirenSpread => 0.29,
            AdvancedStrategyType::BearishTrollSpread => 0.28,
            AdvancedStrategyType::BearishUnicornSpread => 0.27,
            AdvancedStrategyType::BearishValkyriSpread => 0.26,
            AdvancedStrategyType::BearishWendigoSpread => 0.25,
            AdvancedStrategyType::BearishXenosSpread => 0.24,
            AdvancedStrategyType::BearishYetiSpread => 0.23,
            AdvancedStrategyType::BearishZizSpread => 0.22,

            // Neutral Strategies - Base Probabilities
            AdvancedStrategyType::LongIronCondor => 0.70,
            AdvancedStrategyType::ShortIronCondor => 0.65,
            AdvancedStrategyType::LongStrangle => 0.45,
            AdvancedStrategyType::ShortStrangle => 0.55,
            AdvancedStrategyType::LongStraddle => 0.45,
            AdvancedStrategyType::ShortStraddle => 0.55,
            AdvancedStrategyType::BoxSpread => 0.75,
            AdvancedStrategyType::ReverseIronButterfly => 0.60,
            AdvancedStrategyType::DoubleCalendarSpread => 0.65,
            AdvancedStrategyType::DoubleRatioSpread => 0.58,
            AdvancedStrategyType::IronButterfly => 0.62,
            AdvancedStrategyType::ButterflySpread => 0.60,
            AdvancedStrategyType::ChristmasTree => 0.55,
            AdvancedStrategyType::JadeLizard => 0.63,
            AdvancedStrategyType::BrokenWingButterfly => 0.58,
            AdvancedStrategyType::NeutralCalendarSpread => 0.62,
            AdvancedStrategyType::NeutralDiagonalSpread => 0.60,
            AdvancedStrategyType::NeutralDoubleCalendar => 0.64,
            AdvancedStrategyType::NeutralTripleCalendar => 0.66,
            AdvancedStrategyType::NeutralRatioSpread => 0.59,
            AdvancedStrategyType::NeutralDoubleRatio => 0.61,
            AdvancedStrategyType::NeutralTripleRatio => 0.63,
            AdvancedStrategyType::NeutralIronCondorButterfly => 0.65,
            AdvancedStrategyType::NeutralIronAlbatross => 0.62,
            AdvancedStrategyType::NeutralIronPhoenix => 0.64,
            AdvancedStrategyType::NeutralBoxSpread => 0.70,
            AdvancedStrategyType::NeutralTimeSpread => 0.63,
            AdvancedStrategyType::NeutralLeapSpread => 0.65,
            AdvancedStrategyType::NeutralSyntheticStrangle => 0.58,
            AdvancedStrategyType::NeutralSyntheticStraddle => 0.60,
            AdvancedStrategyType::NeutralCollaredStock => 0.67,
            AdvancedStrategyType::NeutralProtectivePut => 0.69,
            AdvancedStrategyType::NeutralCoveredCall => 0.71,
            AdvancedStrategyType::NeutralCoveredStrangle => 0.66,
            AdvancedStrategyType::NeutralCoveredStraddle => 0.64,
            AdvancedStrategyType::NeutralRatioCalendar => 0.62,
            AdvancedStrategyType::NeutralRatioDiagonal => 0.60,
            AdvancedStrategyType::NeutralDoubleRatioCalendar => 0.58,
            AdvancedStrategyType::NeutralTripleRatioDiagonal => 0.56,
            AdvancedStrategyType::NeutralZebraSpread => 0.55,
            AdvancedStrategyType::NeutralButterflySpread => 0.57,
            AdvancedStrategyType::NeutralCondorSpread => 0.59,
            AdvancedStrategyType::NeutralAlbatrossSpread => 0.58,
            AdvancedStrategyType::NeutralSkewedButterfly => 0.56,
            AdvancedStrategyType::NeutralLadderButterfly => 0.54,
            AdvancedStrategyType::NeutralGutStrangle => 0.57,
            AdvancedStrategyType::NeutralStripStrangle => 0.55,
            AdvancedStrategyType::NeutralStrapStrangle => 0.56,
            AdvancedStrategyType::NeutralJadeLizardVariation => 0.61,
            AdvancedStrategyType::NeutralIronCondorVariation => 0.63,
            AdvancedStrategyType::NeutralDragonSpread => 0.64,
            AdvancedStrategyType::NeutralPhoenixSpread => 0.66,
            AdvancedStrategyType::NeutralGriffinSpread => 0.63,
            AdvancedStrategyType::NeutralChimeraSpread => 0.61,
            AdvancedStrategyType::NeutralHydraSpread => 0.59,
            AdvancedStrategyType::NeutralKrakenSpread => 0.58,
            AdvancedStrategyType::NeutralLeviathanSpread => 0.57,
            AdvancedStrategyType::NeutralMantisSpread => 0.56,
            AdvancedStrategyType::NeutralScorpionSpread => 0.55,
            AdvancedStrategyType::NeutralSphinxSpread => 0.54,
            AdvancedStrategyType::NeutralTitanSpread => 0.53,
            AdvancedStrategyType::NeutralVultureSpread => 0.52,
            AdvancedStrategyType::NeutralWyvernSpread => 0.51,
            AdvancedStrategyType::NeutralXiphosSpread => 0.50,
            AdvancedStrategyType::NeutralYggdrasilSpread => 0.49,
            AdvancedStrategyType::NeutralZephyrSpread => 0.48,
            AdvancedStrategyType::NeutralAegisSpread => 0.47,
            AdvancedStrategyType::NeutralBehemothSpread => 0.46,
            AdvancedStrategyType::NeutralColossusSpread => 0.45,
            AdvancedStrategyType::NeutralDryadSpread => 0.44,
            AdvancedStrategyType::NeutralEchoSpread => 0.43,
            AdvancedStrategyType::NeutralFurySpread => 0.42,
            AdvancedStrategyType::NeutralGorgonSpread => 0.41,
            AdvancedStrategyType::NeutralHarpySpread => 0.40,
            AdvancedStrategyType::NeutralImpSpread => 0.39,
            AdvancedStrategyType::NeutralJormungandrSpread => 0.38,
            AdvancedStrategyType::NeutralKirinSpread => 0.37,
            AdvancedStrategyType::NeutralLamiaSpread => 0.36,
            AdvancedStrategyType::NeutralMedusaSpread => 0.35,
            AdvancedStrategyType::NeutralNagaSpread => 0.34,
            AdvancedStrategyType::NeutralOrcSpread => 0.33,
            AdvancedStrategyType::NeutralPegasusSpread => 0.32,
            AdvancedStrategyType::NeutralQuetzalSpread => 0.31,
            AdvancedStrategyType::NeutralRocSpread => 0.30,
            AdvancedStrategyType::NeutralSirenSpread => 0.29,
            AdvancedStrategyType::NeutralTrollSpread => 0.28,
            AdvancedStrategyType::NeutralUnicornSpread => 0.27,
            AdvancedStrategyType::NeutralValkyriSpread => 0.26,
            AdvancedStrategyType::NeutralWendigoSpread => 0.25,
            AdvancedStrategyType::NeutralXenosSpread => 0.24,
            AdvancedStrategyType::NeutralYetiSpread => 0.23,
            AdvancedStrategyType::NeutralZizSpread => 0.22,
        };

        // Adjust probability based on market conditions
        let vol_adjustment = self.calculate_volatility_adjustment();
        let skew_adjustment = self.calculate_skew_adjustment();
        let momentum_adjustment = self.calculate_momentum_adjustment();
        let term_structure_adjustment = self.calculate_term_structure_adjustment();
        let correlation_adjustment = self.calculate_correlation_adjustment();
        
        let final_prob = base_prob 
            * vol_adjustment 
            * skew_adjustment 
            * momentum_adjustment
            * term_structure_adjustment
            * correlation_adjustment;
        
        final_prob.min(0.95).max(0.05) // Cap probability between 5% and 95%
    }

    fn calculate_term_structure_adjustment(&self) -> f64 {
        let term_structure = self.market_data.get_term_structure();
        if term_structure.is_empty() {
            return 1.0;
        }
        
        let slope = self.calculate_term_structure_slope();
        let adjustment = 1.0 + 0.1 * slope;
        
        adjustment.min(1.3).max(0.7)
    }

    fn calculate_correlation_adjustment(&self) -> f64 {
        let correlation = self.market_data.get_correlation_matrix();
        if correlation.is_empty() {
            return 1.0;
        }
        
        let avg_correlation = correlation.iter()
            .flat_map(|row| row.iter())
            .sum::<f64>() / (correlation.len() * correlation[0].len()) as f64;
            
        let adjustment = 1.0 + 0.2 * (avg_correlation - 0.5);
        
        adjustment.min(1.4).max(0.6)
    }
}

#[derive(Debug, Clone)]
struct SkewParameters {
    atm_vol: f64,
    skew: f64,
    convexity: f64,
    term_structure_slope: f64,
    vol_of_vol: f64,
}

#[derive(Debug, Clone)]
enum StrategyError {
    InsufficientData,
    InvalidStrikes,
}
