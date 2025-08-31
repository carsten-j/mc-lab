# Inverse Transform Theory Notebook Requirements

## Overview

Create a comprehensive educational notebook demonstrating the inverse transform method for random number generation, emphasizing theory, mathematical foundations, and visual explanations.

## Target Audience

Students and practitioners learning Monte Carlo methods, requiring both theoretical understanding and practical implementation skills.

## Notebook Name

`inverse_transform_theory_demo.ipynb`

## Learning Objectives

After completing this notebook, readers should:

1. Understand the theoretical foundation of the inverse transform method
2. Know when and why the method works
3. Be able to implement both analytical and numerical versions
4. Appreciate the trade-offs between different sampling methods
5. Visualize how the transformation process works

## Content Requirements

### 1. Theory Section

**Mathematical Foundation:**

- **The Fundamental Theorem**: If U ~ Uniform(0,1) and F is a CDF with inverse F⁻¹, then F⁻¹(U) follows the distribution with CDF F
- Mathematical proof sketch showing:
  - P(F⁻¹(U) ≤ x) = P(U ≤ F(x)) = F(x)
  - Conditions for validity (monotonic CDF)
- Prerequisites: basic probability theory, CDFs, uniform distribution
- Limitations and when the method fails

### 2. Visual Demonstration

**Interactive Visualizations:**

- Step-by-step transformation: uniform samples → target distribution
- Side-by-side plots showing:
  - Uniform random points on [0,1]
  - Corresponding CDF visualization
  - Resulting samples from target distribution
- Multiple distribution examples (exponential, beta, gamma)
- Animation or interactive plots showing the mapping process

### 3. Implementation Comparison - Gamma Distribution

**Target Distribution:** Gamma(shape=2.5, scale=1.5)

- Non-trivial distribution with known properties
- Widely used in practice
- Has both direct sampling and inverse transform options

**Three Methods:**

1. **Direct Sampling** (ground truth): `np.random.gamma(2.5, 1.5)`
2. **Analytical Inverse**: Using `scipy.stats.gamma.ppf()`
3. **Numerical Inverse**: Build inverse CDF using interpolation from existing `mc_lab.inverse_transform`

**Performance Comparison Table:**

| Method | Sample Mean | Sample Std | Theoretical Mean | Theoretical Std | KS Test p-value | Runtime (samples/sec) | Memory Usage |
|--------|-------------|------------|------------------|-----------------|-----------------|-------------------|--------------|
| Direct | ... | ... | 3.75 | 2.74 | ... | ... | ... |
| Analytical | ... | ... | 3.75 | 2.74 | ... | ... | ... |
| Numerical | ... | ... | 3.75 | 2.74 | ... | ... | ... |

### 4. Visual Proof - Why It Works

**Probability Integral Transformation:**

- Show that F(X) ~ Uniform(0,1) when X follows distribution F
- Graphical demonstration of the inverse relationship
- Histogram of F(samples) showing uniform distribution
- QQ plots comparing theoretical vs empirical quantiles

### 5. Advanced Topics

**Using Existing MC-Lab Code:**

- Stratified sampling from `StratifiedInverseTransformSampler`
- Variance reduction demonstration
- Performance considerations and when to use inverse transform vs other methods
- Connection to quasi-Monte Carlo methods

## Technical Requirements

### Code Standards

- Use existing `mc_lab.inverse_transform` module
- Follow MC-Lab's RNG conventions (`RandomState`, `as_generator()`)
- Include proper error handling and input validation
- Use type hints and Google-style docstrings for any helper functions

### Visualization Standards

- Use matplotlib with consistent styling
- Clear axis labels and legends
- Publication-quality figures
- Color-blind friendly palettes
- Interactive elements where appropriate

### Notebook Structure

- Follow MC-Lab notebook guidelines:
  - Tell a story for the audience
  - Document the process, not just results  
  - Use cell divisions to make steps clear
  - Modularize code
  - Design for reading, running, and exploration
- No trailing newlines in notebook files
- Clear markdown explanations between code sections
- Reproducible results with fixed random seeds

### Dependencies

- Only use existing MC-Lab dependencies: NumPy, SciPy, matplotlib, pytest
- Leverage existing `mc_lab.inverse_transform` implementations
- No new package requirements

## Success Criteria

1. **Educational Value**: Clear progression from theory to implementation
2. **Visual Clarity**: Intuitive plots showing the transformation process
3. **Practical Utility**: Useful comparison of methods with performance metrics
4. **Code Quality**: Clean, documented, and following project standards
5. **Reproducibility**: Consistent results across runs with proper seed management

## Deliverables

1. Requirements document (this file) in `docs/` folder
2. Notebook file `notebooks/inverse_transform_theory_demo.ipynb`
3. All code properly integrated with existing MC-Lab infrastructure
