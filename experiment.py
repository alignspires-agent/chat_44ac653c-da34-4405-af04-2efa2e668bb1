
import numpy as np
import sys
import logging
from typing import Tuple, List, Dict, Any
from scipy.stats import norm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LPConformalPrediction:
    """
    Lévy-Prokhorov Robust Conformal Prediction for Time Series with Distribution Shifts.
    Based on the paper: "Conformal Prediction under Lévy–Prokhorov Distribution Shifts:
    Robustness to Local and Global Perturbations" by Aolaritei et al.
    """
    
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, rho: float = 0.05):
        """
        Initialize LP Robust Conformal Prediction model.
        
        Parameters:
        -----------
        alpha : float
            Target miscoverage rate (default: 0.1 for 90% coverage)
        epsilon : float
            Local perturbation parameter for LP ambiguity set
        rho : float
            Global perturbation parameter for LP ambiguity set
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho
        self.calibration_scores = None
        self.quantile = None
        
        logger.info(f"Initialized LP Conformal Prediction with alpha={alpha}, epsilon={epsilon}, rho={rho}")
    
    def fit(self, calibration_scores: np.ndarray) -> None:
        """
        Fit the model using calibration scores.
        
        Parameters:
        -----------
        calibration_scores : np.ndarray
            Array of nonconformity scores from calibration data
        """
        try:
            logger.info("Fitting LP robust conformal prediction model...")
            
            if calibration_scores is None or len(calibration_scores) == 0:
                raise ValueError("Calibration scores cannot be empty")
            
            self.calibration_scores = calibration_scores
            n_calib = len(calibration_scores)
            
            # Calculate adjusted quantile level for finite sample correction
            level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(n_calib))
            
            # Calculate worst-case quantile using LP robustness formula
            # QuantWC_{ε,ρ}(β;P) = Quant(β + ρ; P) + ε
            beta = 1 - self.alpha
            adjusted_beta = min(beta + self.rho, 1.0)  # Ensure beta + ρ ≤ 1
            
            if adjusted_beta >= 1.0:
                logger.warning(f"Adjusted beta {adjusted_beta} >= 1.0, using 0.999")
                adjusted_beta = 0.999
            
            # Calculate empirical quantile
            self.quantile = np.quantile(calibration_scores, adjusted_beta) + self.epsilon
            
            logger.info(f"Fitted model: quantile={self.quantile:.4f}, adjusted_beta={adjusted_beta:.4f}")
            logger.info(f"Calibration scores stats: mean={np.mean(calibration_scores):.4f}, "
                       f"std={np.std(calibration_scores):.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            sys.exit(1)
    
    def predict_interval(self, test_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for test scores.
        
        Parameters:
        -----------
        test_scores : np.ndarray
            Array of nonconformity scores for test data
            
        Returns:
        --------
        Tuple containing:
        - coverage_mask : np.ndarray (boolean array indicating coverage)
        - interval_widths : np.ndarray (width of prediction intervals)
        """
        try:
            if self.quantile is None:
                raise ValueError("Model must be fitted before prediction")
            
            if test_scores is None or len(test_scores) == 0:
                raise ValueError("Test scores cannot be empty")
            
            logger.info("Generating prediction intervals...")
            
            # For each test score, check if it's below the quantile threshold
            coverage_mask = test_scores <= self.quantile
            
            # Calculate coverage statistics
            coverage_rate = np.mean(coverage_mask)
            expected_coverage = 1 - self.alpha
            
            logger.info(f"Prediction intervals generated:")
            logger.info(f"  - Expected coverage: {expected_coverage:.1%}")
            logger.info(f"  - Actual coverage: {coverage_rate:.1%}")
            logger.info(f"  - Quantile threshold: {self.quantile:.4f}")
            logger.info(f"  - Test scores range: [{np.min(test_scores):.4f}, {np.max(test_scores):.4f}]")
            
            # Calculate interval widths (for regression tasks, this would be different)
            # For classification, we return constant width since we're working with scores
            interval_widths = np.full_like(test_scores, self.quantile)
            
            return coverage_mask, interval_widths
            
        except Exception as e:
            logger.error(f"Error generating prediction intervals: {str(e)}")
            sys.exit(1)
    
    def calculate_worst_case_coverage(self, q: float) -> float:
        """
        Calculate worst-case coverage for a given quantile value.
        CovWC_{ε,ρ}(q;P) = F_P(q - ε) - ρ
        
        Parameters:
        -----------
        q : float
            Quantile value
            
        Returns:
        --------
        Worst-case coverage value
        """
        if self.calibration_scores is None:
            raise ValueError("Model must be fitted first")
        
        # Calculate empirical CDF at (q - epsilon)
        empirical_cdf = np.mean(self.calibration_scores <= (q - self.epsilon))
        
        # Apply global perturbation adjustment
        worst_case_coverage = max(empirical_cdf - self.rho, 0.0)
        
        return worst_case_coverage

def generate_time_series_data(n_samples: int = 1000, shift_strength: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with distribution shift.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    shift_strength : float
        Strength of distribution shift (0 = no shift, 1 = strong shift)
        
    Returns:
    --------
    Tuple containing calibration and test scores
    """
    logger.info(f"Generating synthetic time series data with {n_samples} samples...")
    
    # Base distribution (normal)
    base_mean, base_std = 0.0, 1.0
    calibration_scores = np.random.normal(base_mean, base_std, n_samples // 2)
    
    # Introduce distribution shift in test data
    shift_mean = shift_strength * 2.0  # Stronger shift means larger mean difference
    test_scores = np.random.normal(base_mean + shift_mean, base_std * (1 + shift_strength/2), n_samples // 2)
    
    logger.info(f"Data generation complete:")
    logger.info(f"  - Calibration: mean={np.mean(calibration_scores):.4f}, std={np.std(calibration_scores):.4f}")
    logger.info(f"  - Test: mean={np.mean(test_scores):.4f}, std={np.std(test_scores):.4f}")
    logger.info(f"  - Distribution shift strength: {shift_strength}")
    
    return calibration_scores, test_scores

def run_experiment():
    """Main experiment function."""
    logger.info("Starting LP Robust Conformal Prediction Experiment")
    logger.info("=" * 60)
    
    try:
        # Experiment parameters
        n_samples = 1000
        shift_strength = 0.8  # Strong distribution shift
        alpha = 0.1  # 90% target coverage
        epsilon_values = [0.0, 0.1, 0.2]  # Local perturbation parameters
        rho_values = [0.0, 0.05, 0.1]     # Global perturbation parameters
        
        # Generate data with distribution shift
        calibration_scores, test_scores = generate_time_series_data(n_samples, shift_strength)
        
        results = []
        
        # Test different robustness parameter combinations
        for epsilon in epsilon_values:
            for rho in rho_values:
                logger.info(f"\nTesting ε={epsilon}, ρ={rho}")
                logger.info("-" * 30)
                
                # Initialize and fit model
                model = LPConformalPrediction(alpha=alpha, epsilon=epsilon, rho=rho)
                model.fit(calibration_scores)
                
                # Generate prediction intervals
                coverage_mask, interval_widths = model.predict_interval(test_scores)
                coverage_rate = np.mean(coverage_mask)
                
                # Calculate worst-case coverage
                worst_case_cov = model.calculate_worst_case_coverage(model.quantile)
                
                # Store results
                result = {
                    'epsilon': epsilon,
                    'rho': rho,
                    'coverage_rate': coverage_rate,
                    'expected_coverage': 1 - alpha,
                    'quantile': model.quantile,
                    'worst_case_coverage': worst_case_cov,
                    'avg_interval_width': np.mean(interval_widths)
                }
                results.append(result)
                
                logger.info(f"  Coverage: {coverage_rate:.3f} (expected: {1-alpha:.3f})")
                logger.info(f"  Worst-case coverage: {worst_case_cov:.3f}")
                logger.info(f"  Avg interval width: {np.mean(interval_widths):.4f}")
        
        # Print summary of results
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
        for result in results:
            logger.info(f"ε={result['epsilon']:.2f}, ρ={result['rho']:.2f}: "
                       f"Coverage={result['coverage_rate']:.3f}, "
                       f"Width={result['avg_interval_width']:.4f}, "
                       f"Worst-case={result['worst_case_coverage']:.3f}")
        
        # Find best parameter combination
        best_result = min(results, key=lambda x: abs(x['coverage_rate'] - x['expected_coverage']))
        logger.info(f"\nBest parameters: ε={best_result['epsilon']:.2f}, ρ={best_result['rho']:.2f}")
        logger.info(f"Achieved coverage: {best_result['coverage_rate']:.3f} "
                   f"(target: {best_result['expected_coverage']:.3f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("LP Robust Conformal Prediction for Time Series with Distribution Shifts")
    logger.info("Based on: Conformal Prediction under Lévy–Prokhorov Distribution Shifts")
    
    # Run the experiment
    results = run_experiment()
    
    logger.info("Experiment completed successfully!")
