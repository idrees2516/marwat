use super::*;
use ndarray::{Array2, Array1};
use std::collections::HashMap;
use itertools::Itertools;

/// Represents a point on the volatility surface
#[derive(Debug, Clone, Copy)]
pub struct VolPoint {
    pub strike: f64,
    pub time: f64,
    pub volatility: f64,
    pub weight: f64,
}

/// Advanced volatility surface construction and interpolation
pub struct VolatilitySurface {
    points: Vec<VolPoint>,
    strike_grid: Array1<f64>,
    time_grid: Array1<f64>,
    surface: Array2<f64>,
    interpolation_method: InterpolationMethod,
    extrapolation_method: ExtrapolationMethod,
    calibration_weights: HashMap<(f64, f64), f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    Linear,
    Cubic,
    SVI,
    SSVI,
}

#[derive(Debug, Clone, Copy)]
pub enum ExtrapolationMethod {
    Flat,
    Linear,
    StickyStrike,
    StickyDelta,
}

impl VolatilitySurface {
    pub fn new(
        points: Vec<VolPoint>,
        num_strike_points: usize,
        num_time_points: usize,
        interpolation: InterpolationMethod,
        extrapolation: ExtrapolationMethod,
    ) -> Result<Self> {
        if points.is_empty() {
            return Err(PanopticError::InsufficientData);
        }

        // Extract unique sorted strikes and times
        let strikes: Vec<f64> = points.iter()
            .map(|p| p.strike)
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .collect();
        
        let times: Vec<f64> = points.iter()
            .map(|p| p.time)
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .collect();

        // Create uniform grids
        let strike_grid = create_uniform_grid(
            strikes.first().unwrap(),
            strikes.last().unwrap(),
            num_strike_points,
        );
        
        let time_grid = create_uniform_grid(
            times.first().unwrap(),
            times.last().unwrap(),
            num_time_points,
        );

        // Initialize surface with NaN
        let surface = Array2::from_elem(
            (num_strike_points, num_time_points),
            f64::NAN,
        );

        // Initialize calibration weights
        let mut calibration_weights = HashMap::new();
        for point in &points {
            calibration_weights.insert(
                (point.strike, point.time),
                point.weight,
            );
        }

        let mut vol_surface = Self {
            points,
            strike_grid,
            time_grid,
            surface,
            interpolation_method: interpolation,
            extrapolation_method: extrapolation,
            calibration_weights,
        };

        // Construct initial surface
        vol_surface.construct_surface()?;

        Ok(vol_surface)
    }

    /// Construct the volatility surface using the chosen interpolation method
    fn construct_surface(&mut self) -> Result<()> {
        match self.interpolation_method {
            InterpolationMethod::Linear => self.construct_linear_surface(),
            InterpolationMethod::Cubic => self.construct_cubic_surface(),
            InterpolationMethod::SVI => self.construct_svi_surface(),
            InterpolationMethod::SSVI => self.construct_ssvi_surface(),
        }
    }

    /// Get interpolated volatility for any strike and time
    pub fn get_volatility(&self, strike: f64, time: f64) -> Result<f64> {
        // Check if point is within bounds
        if !self.is_point_within_bounds(strike, time) {
            return self.extrapolate_volatility(strike, time);
        }

        // Find nearest grid points
        let (i, i_next, s_ratio) = self.find_nearest_points(
            strike,
            &self.strike_grid,
        )?;
        
        let (j, j_next, t_ratio) = self.find_nearest_points(
            time,
            &self.time_grid,
        )?;

        // Bilinear interpolation
        let v00 = self.surface[[i, j]];
        let v10 = self.surface[[i_next, j]];
        let v01 = self.surface[[i, j_next]];
        let v11 = self.surface[[i_next, j_next]];

        let vol = v00 * (1.0 - s_ratio) * (1.0 - t_ratio) +
                 v10 * s_ratio * (1.0 - t_ratio) +
                 v01 * (1.0 - s_ratio) * t_ratio +
                 v11 * s_ratio * t_ratio;

        Ok(vol)
    }

    /// Extrapolate volatility outside the grid
    fn extrapolate_volatility(&self, strike: f64, time: f64) -> Result<f64> {
        match self.extrapolation_method {
            ExtrapolationMethod::Flat => {
                // Use nearest point on surface
                let strike_clamped = strike.clamp(
                    self.strike_grid[0],
                    *self.strike_grid.last().unwrap(),
                );
                let time_clamped = time.clamp(
                    self.time_grid[0],
                    *self.time_grid.last().unwrap(),
                );
                self.get_volatility(strike_clamped, time_clamped)
            }
            ExtrapolationMethod::Linear => {
                // Linear extrapolation using slope at boundary
                self.linear_extrapolation(strike, time)
            }
            ExtrapolationMethod::StickyStrike => {
                // Use same volatility as nearest expiry
                let time_clamped = time.clamp(
                    self.time_grid[0],
                    *self.time_grid.last().unwrap(),
                );
                self.get_volatility(strike, time_clamped)
            }
            ExtrapolationMethod::StickyDelta => {
                // Maintain constant delta extrapolation
                self.sticky_delta_extrapolation(strike, time)
            }
        }
    }

    /// Construct surface using linear interpolation
    fn construct_linear_surface(&mut self) -> Result<()> {
        for (i, k) in self.strike_grid.iter().enumerate() {
            for (j, t) in self.time_grid.iter().enumerate() {
                // Find nearest points and interpolate
                let vol = self.interpolate_nearest_points(*k, *t)?;
                self.surface[[i, j]] = vol;
            }
        }
        Ok(())
    }

    /// Construct surface using cubic spline interpolation
    fn construct_cubic_surface(&mut self) -> Result<()> {
        // Implementation using cubic splines
        // This would use a crate like 'splines' for cubic interpolation
        unimplemented!()
    }

    /// Construct surface using SVI parameterization
    fn construct_svi_surface(&mut self) -> Result<()> {
        // Implement SVI (Stochastic Volatility Inspired) parameterization
        // This involves fitting SVI parameters for each time slice
        unimplemented!()
    }

    /// Construct surface using SSVI parameterization
    fn construct_ssvi_surface(&mut self) -> Result<()> {
        // Implement Surface SVI parameterization
        // This ensures calendar spread arbitrage freedom
        unimplemented!()
    }

    /// Helper function to find nearest points for interpolation
    fn find_nearest_points(
        &self,
        value: f64,
        grid: &Array1<f64>,
    ) -> Result<(usize, usize, f64)> {
        let n = grid.len();
        
        // Binary search for position
        let mut left = 0;
        let mut right = n - 1;
        
        while left < right {
            let mid = (left + right + 1) / 2;
            if grid[mid] <= value {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        
        let i = left;
        let i_next = (i + 1).min(n - 1);
        
        let ratio = if i == i_next {
            0.0
        } else {
            (value - grid[i]) / (grid[i_next] - grid[i])
        };
        
        Ok((i, i_next, ratio))
    }

    /// Check if point is within surface bounds
    fn is_point_within_bounds(&self, strike: f64, time: f64) -> bool {
        strike >= self.strike_grid[0] &&
        strike <= *self.strike_grid.last().unwrap() &&
        time >= self.time_grid[0] &&
        time <= *self.time_grid.last().unwrap()
    }

    /// Linear extrapolation for points outside grid
    fn linear_extrapolation(&self, strike: f64, time: f64) -> Result<f64> {
        // Implement linear extrapolation using boundary slopes
        unimplemented!()
    }

    /// Sticky delta extrapolation
    fn sticky_delta_extrapolation(&self, strike: f64, time: f64) -> Result<f64> {
        // Implement sticky delta extrapolation
        // This requires computing option deltas and maintaining them
        unimplemented!()
    }

    /// Update surface with new market data
    pub fn update(&mut self, new_points: Vec<VolPoint>) -> Result<()> {
        self.points.extend(new_points);
        self.construct_surface()
    }

    /// Compute the total variance surface
    pub fn total_variance_surface(&self) -> Array2<f64> {
        let mut total_var = Array2::zeros(self.surface.raw_dim());
        for ((i, j), &vol) in self.surface.indexed_iter() {
            total_var[[i, j]] = vol * vol * self.time_grid[j];
        }
        total_var
    }

    /// Get surface arbitrage-free status
    pub fn is_arbitrage_free(&self) -> bool {
        // Check for butterfly and calendar spread arbitrage
        self.check_butterfly_arbitrage() && self.check_calendar_arbitrage()
    }

    /// Check for butterfly arbitrage
    fn check_butterfly_arbitrage(&self) -> bool {
        // Implement butterfly arbitrage check
        // This involves checking convexity in strike direction
        true // Placeholder
    }

    /// Check for calendar spread arbitrage
    fn check_calendar_arbitrage(&self) -> bool {
        // Implement calendar spread arbitrage check
        // This involves checking total variance is increasing in time
        true // Placeholder
    }
}

/// Create a uniform grid between min and max values
fn create_uniform_grid(min: &f64, max: &f64, points: usize) -> Array1<f64> {
    let step = (max - min) / (points - 1) as f64;
    Array1::linspace(*min, *max, points)
}
