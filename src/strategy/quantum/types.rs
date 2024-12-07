use ethers::types::{Address, U256};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionPosition {
    pub token_id: U256,
    pub strike: U256,
    pub maturity: U256,
    pub is_call: bool,
    pub is_long: bool,
    pub size: U256,
    pub collateral: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeggedPosition {
    pub legs: Vec<OptionLeg>,
    pub total_collateral: U256,
    pub strategy_type: StrategyType,
    pub risk_params: RiskParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionLeg {
    pub position: OptionPosition,
    pub ratio: i32,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyType {
    ButterflySpread,
    IronCondor,
    StraddleStrangle,
    CalendarSpread,
    DiagonalSpread,
    RatioSpread,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameters {
    pub max_loss: U256,
    pub max_leverage: f64,
    pub min_collateral_ratio: f64,
    pub greek_limits: GreekLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreekLimits {
    pub max_delta: f64,
    pub max_gamma: f64,
    pub max_vega: f64,
    pub max_theta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeParameters {
    pub delta_threshold: f64,
    pub rebalance_interval: u64,
    pub slippage_tolerance: f64,
    pub gas_price_limit: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub spot_price: U256,
    pub volatility: f64,
    pub interest_rate: f64,
    pub pool_liquidity: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionParams {
    pub max_slippage: f64,
    pub deadline: U256,
    pub min_output: U256,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PricingModel {
    BlackScholes,
    Binomial,
    MonteCarlo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilitySurface {
    pub strike_grid: Vec<U256>,
    pub time_grid: Vec<U256>,
    pub volatilities: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreekSensitivities {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

#[derive(Debug, Clone)]
pub enum AdvancedStrategyType {
    // Bull Strategies (90)
    BullCallSpread,
    BullPutSpread,
    InTheMoneyBullCallSpread,
    DeepInTheMoneyCallSpread,
    LongCallButterfly,
    BullCallLadder,
    BullPutLadder,
    BullishJadeLizard,
    LongCallCondor,
    BullishDiagonalSpread,
    BullishCalendarSpread,
    BullishRatioSpread,
    LongStockSyntheticCall,
    BullishSeagull,
    BullishRiskReversal,
    BullishIronButterfly,
    BullishBrokenWingButterfly,
    BullishSkewedButterfly,
    BullishChristmasTree,
    BullishLadderButterfly,
    BullishDoubleCalendar,
    BullishTripleCalendar,
    BullishQuadCalendar,
    BullishPentaCalendar,
    BullishDoubleRatio,
    BullishTripleRatio,
    BullishQuadRatio,
    BullishPentaRatio,
    BullishBoxSpread,
    BullishStripStrangle,
    BullishStrapStrangle,
    BullishGutStrangle,
    BullishIronAlbatross,
    BullishIronCondorButterfly,
    BullishDiagonalCalendar,
    BullishDiagonalRatio,
    BullishTimeSpread,
    BullishLeapSpread,
    BullishSyntheticStrangle,
    BullishSyntheticStraddle,
    BullishCollaredStock,
    BullishProtectivePut,
    BullishCoveredCall,
    BullishCoveredStrangle,
    BullishCoveredStraddle,
    BullishRatioCalendar,
    BullishRatioDiagonal,
    BullishDoubleRatioCalendar,
    BullishTripleRatioDiagonal,
    BullishZebraSpread,
    BullishButterflySpread,
    BullishCondorSpread,
    BullishAlbatrossSpread,
    BullishIronPhoenix,
    BullishDragonSpread,
    BullishPhoenixSpread,
    BullishGriffinSpread,
    BullishChimeraSpread,
    BullishHydraSpread,
    BullishKrakenSpread,
    BullishLeviathanSpread,
    BullishMantisSpread,
    BullishScorpionSpread,
    BullishSphinxSpread,
    BullishTitanSpread,
    BullishVultureSpread,
    BullishWyvernSpread,
    BullishXiphosSpread,
    BullishYggdrasilSpread,
    BullishZephyrSpread,
    BullishAegisSpread,
    BullishBehemothSpread,
    BullishColossusSpread,
    BullishDryadSpread,
    BullishEchoSpread,
    BullishFurySpread,
    BullishGorgonSpread,
    BullishHarpySpread,
    BullishImpSpread,
    BullishJormungandrSpread,
    BullishKirinSpread,
    BullishLamiaSpread,
    BullishMedusaSpread,
    BullishNagaSpread,
    BullishOrcSpread,
    BullishPegasusSpread,
    BullishQuetzalSpread,
    BullishRocSpread,
    BullishSirenSpread,
    BullishTrollSpread,
    BullishUnicornSpread,
    BullishValkyrieSpread,
    BullishWendigoSpread,
    BullishXenosSpread,
    BullishYetiSpread,
    BullishZizSpread,

    // Bear Strategies (90)
    BearCallSpread,
    BearPutSpread,
    InTheMoneyBearPutSpread,
    DeepInTheMoneyPutSpread,
    ShortCallButterfly,
    BearCallLadder,
    BearPutLadder,
    BearishJadeLizard,
    ShortCallCondor,
    BearishDiagonalSpread,
    BearishCalendarSpread,
    BearishRatioSpread,
    ShortStockSyntheticPut,
    BearishSeagull,
    BearishRiskReversal,
    BearishIronButterfly,
    BearishBrokenWingButterfly,
    BearishSkewedButterfly,
    BearishChristmasTree,
    BearishLadderButterfly,
    BearishDoubleCalendar,
    BearishTripleCalendar,
    BearishQuadCalendar,
    BearishPentaCalendar,
    BearishDoubleRatio,
    BearishTripleRatio,
    BearishQuadRatio,
    BearishPentaRatio,
    BearishBoxSpread,
    BearishStripStrangle,
    BearishStrapStrangle,
    BearishGutStrangle,
    BearishIronAlbatross,
    BearishIronCondorButterfly,
    BearishDiagonalCalendar,
    BearishDiagonalRatio,
    BearishTimeSpread,
    BearishLeapSpread,
    BearishSyntheticStrangle,
    BearishSyntheticStraddle,
    BearishCollaredStock,
    BearishProtectivePut,
    BearishCoveredCall,
    BearishCoveredStrangle,
    BearishCoveredStraddle,
    BearishRatioCalendar,
    BearishRatioDiagonal,
    BearishDoubleRatioCalendar,
    BearishTripleRatioDiagonal,
    BearishZebraSpread,
    BearishButterflySpread,
    BearishCondorSpread,
    BearishAlbatrossSpread,
    BearishIronPhoenix,
    BearishDragonSpread,
    BearishPhoenixSpread,
    BearishGriffinSpread,
    BearishChimeraSpread,
    BearishHydraSpread,
    BearishKrakenSpread,
    BearishLeviathanSpread,
    BearishMantisSpread,
    BearishScorpionSpread,
    BearishSphinxSpread,
    BearishTitanSpread,
    BearishVultureSpread,
    BearishWyvernSpread,
    BearishXiphosSpread,
    BearishYggdrasilSpread,
    BearishZephyrSpread,
    BearishAegisSpread,
    BearishBehemothSpread,
    BearishColossusSpread,
    BearishDryadSpread,
    BearishEchoSpread,
    BearishFurySpread,
    BearishGorgonSpread,
    BearishHarpySpread,
    BearishImpSpread,
    BearishJormungandrSpread,
    BearishKirinSpread,
    BearishLamiaSpread,
    BearishMedusaSpread,
    BearishNagaSpread,
    BearishOrcSpread,
    BearishPegasusSpread,
    BearishQuetzalSpread,
    BearishRocSpread,
    BearishSirenSpread,
    BearishTrollSpread,
    BearishUnicornSpread,
    BearishValkyriSpread,
    BearishWendigoSpread,
    BearishXenosSpread,
    BearishYetiSpread,
    BearishZizSpread,

    // Neutral Strategies (90)
    LongIronCondor,
    ShortIronCondor,
    LongStrangle,
    ShortStrangle,
    LongStraddle,
    ShortStraddle,
    BoxSpread,
    ReverseIronButterfly,
    DoubleCalendarSpread,
    DoubleRatioSpread,
    IronButterfly,
    ButterflySpread,
    ChristmasTree,
    JadeLizard,
    BrokenWingButterfly,
    NeutralCalendarSpread,
    NeutralDiagonalSpread,
    NeutralDoubleCalendar,
    NeutralTripleCalendar,
    NeutralQuadCalendar,
    NeutralPentaCalendar,
    NeutralRatioSpread,
    NeutralDoubleRatio,
    NeutralTripleRatio,
    NeutralQuadRatio,
    NeutralPentaRatio,
    NeutralIronCondorButterfly,
    NeutralIronAlbatross,
    NeutralIronPhoenix,
    NeutralBoxSpread,
    NeutralTimeSpread,
    NeutralLeapSpread,
    NeutralSyntheticStrangle,
    NeutralSyntheticStraddle,
    NeutralCollaredStock,
    NeutralProtectivePut,
    NeutralCoveredCall,
    NeutralCoveredStrangle,
    NeutralCoveredStraddle,
    NeutralRatioCalendar,
    NeutralRatioDiagonal,
    NeutralDoubleRatioCalendar,
    NeutralTripleRatioDiagonal,
    NeutralZebraSpread,
    NeutralButterflySpread,
    NeutralCondorSpread,
    NeutralAlbatrossSpread,
    NeutralSkewedButterfly,
    NeutralLadderButterfly,
    NeutralGutStrangle,
    NeutralStripStrangle,
    NeutralStrapStrangle,
    NeutralJadeLizardVariation,
    NeutralIronCondorVariation,
    NeutralDragonSpread,
    NeutralPhoenixSpread,
    NeutralGriffinSpread,
    NeutralChimeraSpread,
    NeutralHydraSpread,
    NeutralKrakenSpread,
    NeutralLeviathanSpread,
    NeutralMantisSpread,
    NeutralScorpionSpread,
    NeutralSphinxSpread,
    NeutralTitanSpread,
    NeutralVultureSpread,
    NeutralWyvernSpread,
    NeutralXiphosSpread,
    NeutralYggdrasilSpread,
    NeutralZephyrSpread,
    NeutralAegisSpread,
    NeutralBehemothSpread,
    NeutralColossusSpread,
    NeutralDryadSpread,
    NeutralEchoSpread,
    NeutralFurySpread,
    NeutralGorgonSpread,
    NeutralHarpySpread,
    NeutralImpSpread,
    NeutralJormungandrSpread,
    NeutralKirinSpread,
    NeutralLamiaSpread,
    NeutralMedusaSpread,
    NeutralNagaSpread,
    NeutralOrcSpread,
    NeutralPegasusSpread,
    NeutralQuetzalSpread,
    NeutralRocSpread,
    NeutralSirenSpread,
    NeutralTrollSpread,
    NeutralUnicornSpread,
    NeutralValkyriSpread,
    NeutralWendigoSpread,
    NeutralXenosSpread,
    NeutralYetiSpread,
    NeutralZizSpread,
}

#[derive(Debug, Clone)]
pub struct VolatilitySurfaceAdvanced {
    pub term_structure: Vec<f64>,
    pub strike_structure: Vec<f64>,
    pub implied_volatilities: Vec<Vec<f64>>,
    pub skew_parameters: SkewParameters,
    pub surface_interpolation: SurfaceInterpolation,
}

#[derive(Debug, Clone)]
pub struct SkewParameters {
    pub atm_vol: f64,
    pub skew: f64,
    pub convexity: f64,
    pub term_structure_slope: f64,
    pub vol_of_vol: f64,
}

#[derive(Debug, Clone)]
pub struct SurfaceInterpolation {
    pub method: InterpolationMethod,
    pub tension: f64,
    pub smoothing: f64,
}

#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    CubicSpline,
    BilinearInterpolation,
    BicubicInterpolation,
    SabrModel,
}

#[derive(Debug, Clone)]
pub struct StrategyParameters {
    pub strategy_type: AdvancedStrategyType,
    pub position_sizing: PositionSizing,
    pub risk_parameters: RiskParameters,
    pub optimization_constraints: OptimizationConstraints,
    pub execution_parameters: ExecutionParameters,
}

#[derive(Debug, Clone)]
pub struct PositionSizing {
    pub kelly_fraction: f64,
    pub max_position_size: f64,
    pub position_scaling: PositionScaling,
    pub risk_adjusted_sizing: bool,
}

#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    pub max_loss: f64,
    pub min_profit_target: f64,
    pub max_vega_exposure: f64,
    pub max_theta_exposure: f64,
    pub position_limits: PositionLimits,
}

#[derive(Debug, Clone)]
pub struct ExecutionParameters {
    pub slippage_model: SlippageModel,
    pub execution_style: ExecutionStyle,
    pub rebalance_triggers: Vec<RebalanceTrigger>,
}

#[derive(Debug, Clone)]
pub enum SlippageModel {
    ConstantSlippage,
    ProportionalSlippage,
}

#[derive(Debug, Clone)]
pub enum ExecutionStyle {
    MarketOrder,
    LimitOrder,
    StopOrder,
}

#[derive(Debug, Clone)]
pub enum RebalanceTrigger {
    TimeBased,
    PriceBased,
    VolatilityBased,
}

#[derive(Debug, Clone)]
pub enum PositionScaling {
    LinearScaling,
    NonLinearScaling,
}

#[derive(Debug, Clone)]
pub struct PositionLimits {
    pub max_position_size: f64,
    pub min_position_size: f64,
}
