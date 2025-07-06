import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def check_gaussian(data, plot=True):
    """
    Check if input data follows a Gaussian distribution.
    
    Parameters:
    - data: List or NumPy array of numerical data
    - plot: Boolean, whether to display histogram and Q-Q plot (default: True)
    
    Returns:
    - dict: Contains test results (Shapiro-Wilk stat, p-value, skewness, kurtosis, interpretation)
    """
    # Convert input to NumPy array
    data = np.array(data)
    
    # Initialize results dictionary
    results = {}
    
    # 1. Descriptive Statistics
    results['skewness'] = stats.skew(data)
    results['kurtosis'] = stats.kurtosis(data)
    
    # 2. Shapiro-Wilk Test
    stat, p = stats.shapiro(data)
    results['shapiro_stat'] = stat
    results['shapiro_p'] = p
    results['interpretation'] = ("Data appears Gaussian (fail to reject H0)" if p > 0.05 
                                 else "Data does not appear Gaussian (reject H0)")
    results['interpretation_binary'] = 1 if p > 0.05 else 0
    
    # 3. Plots (if enabled)
    if plot:
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, density=True, alpha=0.7, color='blue')
        plt.title('Histogram of Data')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        # Q-Q Plot
        plt.subplot(1, 2, 2)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()
    
    return results

if __name__ == "__main__":
    sample_data = np.random.normal(loc=0, scale=1, size=1000)  # Gaussian data
    results = check_gaussian(sample_data, plot=True)
    print("Shapiro-Wilk Test:")
    print(f"Statistic: {results['shapiro_stat']:.3f}, p-value: {results['shapiro_p']:.3f}")
    print(f"Interpretation: {results['interpretation']}")
    print(f"Skewness: {results['skewness']:.3f}")
    print(f"Kurtosis: {results['kurtosis']:.3f}")

    sample_data = np.random.exponential(scale=1, size=1000)  # Non-Gaussian data
    results = check_gaussian(sample_data, plot=True)
    print("Exponential Test:")
    print(f"Statistic: {results['shapiro_stat']:.3f}, p-value: {results['shapiro_p']:.3f}")
    print(f"Interpretation: {results['interpretation']}")
    print(f"Skewness: {results['skewness']:.3f}")
    print(f"Kurtosis: {results['kurtosis']:.3f}")