#!/usr/bin/env python3
"""
Validation script to test that the specific scipy import error is resolved.
"""

import os
import subprocess
import sys
import tempfile


def test_scipy_import():
    """Test the specific scipy import that was failing in Colab"""

    print("🧪 Testing scipy import fix...")

    test_script = """
import subprocess
import sys

print("📦 Installing mc-lab-edu==0.2.2...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "mc-lab-edu==0.2.2"], 
                      capture_output=True, text=True)

if result.returncode != 0:
    print("❌ Installation failed:")
    print(result.stderr)
    sys.exit(1)

print("✅ Installation successful!")

# Test the specific imports that were failing
try:
    print("🔍 Testing matplotlib import...")
    import matplotlib.pyplot as plt
    print("✅ matplotlib.pyplot imported successfully")
    
    print("🔍 Testing numpy import...")
    import numpy as np
    print(f"✅ numpy imported successfully (version: {np.__version__})")
    
    print("🔍 Testing scipy.stats import (the one that was failing)...")
    from scipy import stats
    print("✅ scipy.stats imported successfully")
    
    print("🔍 Testing mc_lab functionality...")
    from mc_lab import inverse_transform, rejection_sampling
    print("✅ mc_lab imports successful")
    
    # Test basic functionality
    from mc_lab.inverse_transform import InverseTransformSampler
    sampler = InverseTransformSampler(lambda u: -np.log(1-u))
    samples = sampler.sample(100)
    print(f"✅ Generated {len(samples)} samples successfully")
    
    print("🎉 All tests passed! The fix works correctly.")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during testing: {e}")
    sys.exit(1)
"""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "test_imports.py")
        with open(script_path, "w") as f:
            f.write(test_script)

        # Run the test script
        print("🔄 Running import test in clean environment...")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=180,
            )

            print("📋 Output:")
            print(result.stdout)

            if result.stderr:
                print("⚠️  Warnings/Errors:")
                print(result.stderr)

            if result.returncode == 0:
                print("✅ Validation passed! The scipy import should work in Colab.")
                return True
            else:
                print("❌ Validation failed!")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Test timed out after 180 seconds")
            return False
        except Exception as e:
            print(f"❌ Error running validation: {e}")
            return False


if __name__ == "__main__":
    success = test_scipy_import()
    sys.exit(0 if success else 1)
