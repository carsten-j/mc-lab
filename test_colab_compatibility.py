#!/usr/bin/env python3
"""
Test script to simulate what happens when running the notebook in Google Colab
"""

import os
import subprocess
import sys
import tempfile


def test_colab_compatibility():
    """Test that mc-lab-edu installs without conflicts in a clean environment"""

    print("🧪 Testing Colab compatibility...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test script that simulates the notebook installation
        test_script = """
import importlib.util
import subprocess
import sys

# Check if mc_lab package is available
spec = importlib.util.find_spec("mc_lab")

if spec is None:
    print("📦 Installing mc-lab-edu with numpy compatibility for Colab...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "mc-lab-edu==0.2.1"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ mc-lab-edu has been installed successfully!")
    else:
        print("❌ Installation failed:")
        print(result.stderr)
        sys.exit(1)
else:
    print("✅ mc-lab-edu is already installed")

# Test imports
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mc_lab import inverse_transform, rejection_sampling
    print(f"✅ All imports successful!")
    print(f"📊 NumPy version: {np.__version__}")
    
    # Test basic functionality
    from mc_lab.inverse_transform import InverseTransformSampler
    sampler = InverseTransformSampler(lambda u: -np.log(1-u))
    samples = sampler.sample(100)
    print(f"✅ Generated {len(samples)} samples successfully")
    
    print("🎉 Ready to use mc-lab!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during testing: {e}")
    sys.exit(1)
"""

        script_path = os.path.join(temp_dir, "test_install.py")
        with open(script_path, "w") as f:
            f.write(test_script)

        # Run the test script in a clean Python environment
        print("🔄 Running test in clean environment...")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            print("📋 Output:")
            print(result.stdout)

            if result.stderr:
                print("⚠️  Warnings/Errors:")
                print(result.stderr)

            if result.returncode == 0:
                print("✅ Test passed! The package should work in Colab.")
                return True
            else:
                print("❌ Test failed!")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Test timed out after 120 seconds")
            return False
        except Exception as e:
            print(f"❌ Error running test: {e}")
            return False


if __name__ == "__main__":
    success = test_colab_compatibility()
    sys.exit(0 if success else 1)
