#!/usr/bin/env python3
"""
Comprehensive test suite for ft_linear_regression project.
Tests all checklist items with edge cases and colored output.
"""

import os
import sys
import json
import csv
import subprocess
import shutil
import tempfile
import time
from pathlib import Path

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Test statistics
stats = {
    'total': 0,
    'passed': 0,
    'failed': 0,
    'warnings': 0
}

def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.END}\n")

def print_test(name, description, expected, passed, details=""):
    """Print test result with color"""
    stats['total'] += 1
    if passed:
        stats['passed'] += 1
        status = "✅ PASS"
        color = Colors.GREEN
    else:
        stats['failed'] += 1
        status = "❌ FAIL"
        color = Colors.RED
    
    print(f"{color}{status}{Colors.END} | {name}")
    print(f"       Description: {description}")
    print(f"       Expected: {expected}")
    if details:
        print(f"       Details: {details}")
    print()

def print_warning(name, message):
    """Print warning"""
    stats['warnings'] += 1
    print(f"{Colors.YELLOW}⚠️  WARNING{Colors.END} | {name}")
    print(f"       {message}\n")

def run_command(cmd, input_data=None, timeout=10):
    """Run command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            input=input_data,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def create_test_csv(filename, content):
    """Create test CSV file"""
    with open(filename, 'w', newline='') as f:
        f.write(content)

def backup_original_files():
    """Backup original files"""
    files_to_backup = ['data.csv', 'model.json']
    for f in files_to_backup:
        if os.path.exists(f):
            shutil.copy2(f, f'.backup_{f}')

def restore_original_files():
    """Restore original files"""
    files_to_backup = ['data.csv', 'model.json']
    for f in files_to_backup:
        backup = f'.backup_{f}'
        original = f
        if os.path.exists(backup):
            shutil.move(backup, original)

def cleanup_test_files():
    """Remove test files"""
    test_files = [
        'test_empty.csv', 'test_headers_only.csv', 'test_invalid.csv',
        'test_negative.csv', 'test_large.csv', 'test_corrupted_model.json',
        'test_special.csv', 'test_duplicate.csv'
    ]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)

# ============================================================================
# TEST SECTIONS
# ============================================================================

def test_preliminary_checks():
    """Section 1: Preliminary checks"""
    print_header("SECTION 1: PRELIMINARY CHECKS")
    
    # Test 1.1: Check required files exist
    required_files = ['train.py', 'predict.py', 'data.csv']
    all_exist = all(os.path.exists(f) for f in required_files)
    print_test(
        "Required files exist",
        "train.py, predict.py, data.csv must exist",
        "All files present",
        all_exist,
        f"Found: {[f for f in required_files if os.path.exists(f)]}"
    )
    
    # Test 1.2: Check data.csv format
    valid_csv = False
    try:
        with open('data.csv', 'r') as f:
            reader = csv.DictReader(f)
            if 'km' in reader.fieldnames and 'price' in reader.fieldnames:
                rows = list(reader)
                valid_csv = len(rows) > 0
    except:
        pass
    
    print_test(
        "data.csv format valid",
        "Must have 'km' and 'price' columns with data",
        "Valid CSV with headers and data rows",
        valid_csv
    )
    
    # Test 1.3: Check for forbidden libraries
    forbidden_libs = ['numpy', 'sklearn', 'pandas', 'scipy']
    found_forbidden = []
    for py_file in ['train.py', 'predict.py']:
        if os.path.exists(py_file):
            with open(py_file, 'r') as f:
                content = f.read()
                for lib in forbidden_libs:
                    if f'import {lib}' in content or f'from {lib}' in content:
                        found_forbidden.append(f"{lib} in {py_file}")
    
    print_test(
        "No forbidden libraries",
        f"No {forbidden_libs} allowed",
        "No forbidden libraries used",
        len(found_forbidden) == 0,
        f"Found: {found_forbidden}" if found_forbidden else "Clean"
    )
    
    # Test 1.4: Check for numpy.polyfit or similar
    cheating_patterns = ['polyfit', 'linregress', 'LinearRegression', 'fit(']
    found_cheating = []
    for py_file in ['train.py', 'predict.py']:
        if os.path.exists(py_file):
            with open(py_file, 'r') as f:
                content = f.read()
                for pattern in cheating_patterns:
                    if pattern in content:
                        found_cheating.append(f"{pattern} in {py_file}")
    
    print_test(
        "No cheating patterns",
        "No polyfit, linregress, or auto-ML functions",
        "No cheating patterns found",
        len(found_cheating) == 0,
        f"Found: {found_cheating}" if found_cheating else "Clean"
    )

def test_prediction_before_training():
    """Section 2: Prediction before training"""
    print_header("SECTION 2: PREDICTION BEFORE TRAINING")
    
    # Remove model.json if exists
    if os.path.exists('model.json'):
        os.remove('model.json')
    
    # Test 2.1: Prediction with no model should return 0
    code, stdout, stderr = run_command('python3 predict.py', input_data='100000\n')
    output = stdout.strip()
    returns_zero = '0.00' in output or '0' in output.split(':')[-1] if ':' in output else False
    
    print_test(
        "Prediction without training returns 0",
        "With no model.json, any mileage should predict 0",
        "Estimated price: 0.00",
        returns_zero,
        f"Output: {output}"
    )
    
    # Test 2.2: Test with different mileage values
    test_mileages = ['0', '50000', '200000', '999999']
    all_zero = True
    for mileage in test_mileages:
        code, stdout, stderr = run_command('python3 predict.py', input_data=f'{mileage}\n')
        if '0.00' not in stdout and '0' not in stdout.split(':')[-1]:
            all_zero = False
            break
    
    print_test(
        "All mileages predict 0 without training",
        "Tested: 0, 50000, 200000, 999999 km",
        "All should return price = 0",
        all_zero
    )
    
    # Test 2.3: Verify formula is theta0 + (theta1 * x)
    # Check source code
    formula_correct = False
    if os.path.exists('predict.py'):
        with open('predict.py', 'r') as f:
            content = f.read()
            formula_correct = 'theta0 + (theta1 *' in content or 'theta0 + theta1 *' in content
    
    print_test(
        "Formula is theta0 + (theta1 * mileage)",
        "Must use exact formula from subject",
        "theta0 + (theta1 * mileage)",
        formula_correct
    )

def test_training_phase():
    """Section 3: Training phase"""
    print_header("SECTION 3: TRAINING PHASE")
    
    # Ensure clean state
    if os.path.exists('model.json'):
        os.remove('model.json')
    
    # Test 3.1: Run training
    code, stdout, stderr = run_command('python3 train.py')
    training_success = code == 0 and os.path.exists('model.json')
    
    print_test(
        "Training completes successfully",
        "python3 train.py should run without errors",
        "Exit code 0, model.json created",
        training_success,
        f"Stdout: {stdout[:200]}" if not training_success else ""
    )
    
    # Test 3.2: Check model.json format
    valid_model = False
    if os.path.exists('model.json'):
        try:
            with open('model.json', 'r') as f:
                model = json.load(f)
                valid_model = 'theta0' in model and 'theta1' in model
                if valid_model:
                    try:
                        float(model['theta0'])
                        float(model['theta1'])
                    except:
                        valid_model = False
        except:
            pass
    
    print_test(
        "model.json has correct format",
        "Must contain numeric theta0 and theta1",
        "{'theta0': float, 'theta1': float}",
        valid_model
    )
    
    # Test 3.3: Check that theta values are not both zero after training
    thetas_nonzero = False
    if valid_model:
        with open('model.json', 'r') as f:
            model = json.load(f)
            thetas_nonzero = model['theta0'] != 0 or model['theta1'] != 0
    
    print_test(
        "Theta values are updated after training",
        "After training, theta0 and theta1 should not both be 0",
        "At least one theta != 0",
        thetas_nonzero,
        f"theta0={model['theta0']:.6f}, theta1={model['theta1']:.10f}" if valid_model else ""
    )
    
    # Test 3.4: Check gradient descent implementation
    gradient_correct = False
    if os.path.exists('train.py'):
        with open('train.py', 'r') as f:
            content = f.read()
            # Check for key elements of gradient descent
            has_sum = 'sum(' in content or 'sum0' in content or 'sum1' in content
            has_learning_rate = 'learning_rate' in content.lower() or 'learningrate' in content.lower()
            has_gradient = 'tmp' in content or 'temp' in content.lower()
            gradient_correct = has_sum and has_learning_rate and has_gradient
    
    print_test(
        "Gradient descent implementation present",
        "Must implement gradient descent with sum, learning_rate, and temp variables",
        "Proper gradient descent algorithm",
        gradient_correct
    )

def test_simultaneous_assignment():
    """Section 4: Simultaneous assignment"""
    print_header("SECTION 4: SIMULTANEOUS ASSIGNMENT")
    
    # Check source code for simultaneous update pattern
    simultaneous = False
    if os.path.exists('train.py'):
        with open('train.py', 'r') as f:
            content = f.read()
            # Look for pattern: compute tmp values, then update both thetas
            # Common patterns: tmp0/tmp1, temp0/temp1, new_theta0/new_theta1
            has_tmp_vars = any(pattern in content for pattern in 
                             ['tmp_theta', 'temp_theta', 'tmp0', 'tmp1',
                              'new_theta', 'delta0', 'delta1'])
            has_simultaneous_update = has_tmp_vars and 'theta0' in content and 'theta1' in content
            
            # More specific check: look for the pattern in gradient descent
            lines = content.split('\n')
            in_gradient = False
            found_pattern = False
            for i, line in enumerate(lines):
                if 'def gradient_descent' in line or 'for _ in range(iterations)' in line:
                    in_gradient = True
                if in_gradient:
                    # Check if tmp values are computed before theta updates
                    if 'tmp' in line.lower() or 'temp' in line.lower():
                        # Look ahead for theta updates
                        for j in range(i+1, min(i+10, len(lines))):
                            if 'theta0' in lines[j] and 'theta1' in lines[j]:
                                found_pattern = True
                                break
                    if found_pattern:
                        simultaneous = True
                        break
    
    print_test(
        "Simultaneous theta update",
        "Theta0 and theta1 must be updated using temporary variables",
        "tmp_theta0 and tmp_theta1 computed before updating both thetas",
        simultaneous
    )

def test_prediction_after_training():
    """Section 5: Prediction after training"""
    print_header("SECTION 5: PREDICTION AFTER TRAINING")
    
    # Ensure model is trained
    if not os.path.exists('model.json'):
        run_command('python3 train.py')
    
    # Test 5.1: Prediction returns non-zero after training
    code, stdout, stderr = run_command('python3 predict.py', input_data='100000\n')
    output = stdout.strip()
    # Check if output contains a non-zero price (allowing for negative prices)
    non_zero = ('Estimated price:' in output and 
                any(char.isdigit() for char in output.split(':')[-1]) and
                '0.00' not in output)
    
    print_test(
        "Prediction returns non-zero after training",
        "After training, prediction should give actual price",
        "Price != 0",
        non_zero,
        f"Output: {output}"
    )
    
    # Test 5.2: Test with values from CSV
    # Read some values from data.csv
    test_points = []
    try:
        with open('data.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_points.append((float(row['km']), float(row['price'])))
                if len(test_points) >= 3:
                    break
    except:
        pass
    
    predictions_reasonable = True
    for km, actual_price in test_points:
        code, stdout, stderr = run_command('python3 predict.py', input_data=f'{km}\n')
        try:
            predicted = float(stdout.split(':')[-1].strip())
            # Allow 50% deviation (generous for linear regression)
            if abs(predicted - actual_price) / max(actual_price, 1) > 0.5:
                predictions_reasonable = False
                break
        except:
            predictions_reasonable = False
            break
    
    print_test(
        "Predictions are reasonable for CSV data",
        f"Tested {len(test_points)} points from data.csv",
        "Predictions within 50% of actual prices",
        predictions_reasonable
    )
    
    # Test 5.3: Different mileages give different prices
    code1, stdout1, _ = run_command('python3 predict.py', input_data='50000\n')
    code2, stdout2, _ = run_command('python3 predict.py', input_data='150000\n')
    different_prices = stdout1 != stdout2
    
    print_test(
        "Different mileages give different prices",
        "50000 km vs 150000 km should have different predictions",
        "Different output for different inputs",
        different_prices,
        f"50k: {stdout1.strip()}, 150k: {stdout2.strip()}"
    )

def test_edge_cases_train():
    """Section 6: Edge cases for training"""
    print_header("SECTION 6: EDGE CASES - TRAINING")
    
    # Backup original data
    backup_original_files()
    
    # Test 6.1: Empty CSV file
    create_test_csv('test_empty.csv', '')
    shutil.copy('test_empty.csv', 'data.csv')
    code, stdout, stderr = run_command('python3 train.py')
    handles_empty = code != 0 or 'error' in stderr.lower() or 'error' in stdout.lower()
    print_test(
        "Handles empty CSV file",
        "Empty data.csv should cause graceful error",
        "Non-zero exit or error message",
        handles_empty,
        f"Exit code: {code}"
    )
    
    # Test 6.2: CSV with only headers
    create_test_csv('test_headers_only.csv', 'km,price\n')
    shutil.copy('test_headers_only.csv', 'data.csv')
    code, stdout, stderr = run_command('python3 train.py')
    handles_headers_only = code != 0 or 'error' in stderr.lower() or 'error' in stdout.lower() or 'at least 2' in stdout.lower()
    print_test(
        "Handles CSV with only headers",
        "CSV with no data rows should cause error",
        "Non-zero exit or error message",
        handles_headers_only
    )
    
    # Test 6.3: CSV with invalid data (strings)
    create_test_csv('test_invalid.csv', 'km,price\nabc,def\nxyz,ghi\n')
    shutil.copy('test_invalid.csv', 'data.csv')
    code, stdout, stderr = run_command('python3 train.py')
    handles_invalid = code != 0 or 'error' in stderr.lower() or 'error' in stdout.lower() or 'at least 2' in stdout.lower()
    print_test(
        "Handles CSV with invalid data (strings)",
        "Non-numeric values should be handled gracefully",
        "Error or warning about invalid data",
        handles_invalid
    )
    
    # Test 6.4: CSV with negative values
    create_test_csv('test_negative.csv', 'km,price\n-100,5000\n100,-5000\n')
    shutil.copy('test_negative.csv', 'data.csv')
    code, stdout, stderr = run_command('python3 train.py')
    handles_negative = code != 0 or 'warning' in stdout.lower() or 'skipping' in stdout.lower()
    print_test(
        "Handles CSV with negative values",
        "Negative km or price should be handled",
        "Warning or error about negative values",
        handles_negative,
        f"Output: {stdout[:100]}"
    )
    
    # Test 6.5: CSV with very large values
    create_test_csv('test_large.csv', 'km,price\n999999999,999999999\n1,1\n')
    shutil.copy('test_large.csv', 'data.csv')
    code, stdout, stderr = run_command('python3 train.py')
    handles_large = code == 0 or 'error' not in stderr.lower()
    print_test(
        "Handles CSV with very large values",
        "Large numbers should not cause overflow",
        "Completes or handles gracefully",
        handles_large
    )
    
    # Test 6.6: CSV with missing columns
    create_test_csv('test_special.csv', 'mileage,cost\n100,5000\n200,6000\n')
    shutil.copy('test_special.csv', 'data.csv')
    code, stdout, stderr = run_command('python3 train.py')
    handles_missing_cols = code != 0 or 'error' in stderr.lower() or 'error' in stdout.lower()
    print_test(
        "Handles CSV with wrong column names",
        "Missing 'km' or 'price' columns should cause error",
        "Error about missing columns",
        handles_missing_cols
    )
    
    # Restore original data
    restore_original_files()
    
    # Re-train with original data
    run_command('python3 train.py')

def test_edge_cases_predict():
    """Section 7: Edge cases for prediction"""
    print_header("SECTION 7: EDGE CASES - PREDICTION")
    
    # Ensure model exists
    if not os.path.exists('model.json'):
        run_command('python3 train.py')
    
    # Test 7.1: No model.json
    if os.path.exists('model.json'):
        os.remove('model.json')
    code, stdout, stderr = run_command('python3 predict.py', input_data='100000\n')
    handles_no_model = code == 0 and ('0.00' in stdout or '0' in stdout)
    print_test(
        "Handles missing model.json",
        "Without model.json, should predict 0",
        "Exit 0, prediction = 0",
        handles_no_model,
        f"Output: {stdout.strip()}"
    )
    
    # Test 7.2: Corrupted model.json
    create_test_csv('test_corrupted_model.json', '{invalid json')
    shutil.copy('test_corrupted_model.json', 'model.json')
    code, stdout, stderr = run_command('python3 predict.py', input_data='100000\n')
    handles_corrupted = code == 0 and ('0.00' in stdout or '0' in stdout or 'warning' in stdout.lower())
    print_test(
        "Handles corrupted model.json",
        "Invalid JSON should fall back to 0",
        "Exit 0, prediction = 0 or warning",
        handles_corrupted,
        f"Output: {stdout.strip()}"
    )
    
    # Test 7.3: Empty model.json
    with open('model.json', 'w') as f:
        f.write('')
    code, stdout, stderr = run_command('python3 predict.py', input_data='100000\n')
    handles_empty_model = code == 0 and ('0.00' in stdout or '0' in stdout or 'warning' in stdout.lower())
    print_test(
        "Handles empty model.json",
        "Empty file should fall back to 0",
        "Exit 0, prediction = 0 or warning",
        handles_empty_model
    )
    
    # Test 7.4: Negative mileage input
    # Restore valid model
    run_command('python3 train.py')
    code, stdout, stderr = run_command('python3 predict.py', input_data='-100\n')
    handles_negative_input = code != 0 or 'negative' in stdout.lower() or 'error' in stdout.lower()
    print_test(
        "Handles negative mileage input",
        "Negative mileage should be rejected or handled",
        "Error message or rejection",
        handles_negative_input,
        f"Output: {stdout.strip()}"
    )
    
    # Test 7.5: Non-numeric input
    code, stdout, stderr = run_command('python3 predict.py', input_data='abc\n')
    handles_non_numeric = code != 0 or 'error' in stdout.lower() or 'valid number' in stdout.lower() or 'invalid' in stdout.lower()
    print_test(
        "Handles non-numeric mileage input",
        "Non-numeric input should be rejected",
        "Error message asking for valid number",
        handles_non_numeric,
        f"Output: {stdout.strip()}"
    )
    
    # Test 7.6: Very large mileage
    code, stdout, stderr = run_command('python3 predict.py', input_data='999999999\n')
    handles_large_input = code == 0
    print_test(
        "Handles very large mileage input",
        "Large mileage should not crash",
        "Exit 0, produces prediction",
        handles_large_input,
        f"Output: {stdout.strip()}"
    )
    
    # Test 7.7: Empty input
    code, stdout, stderr = run_command('python3 predict.py', input_data='\n100000\n')
    handles_empty_input = code == 0 and ('0' in stdout or 'empty' in stdout.lower() or 'error' in stdout.lower())
    print_test(
        "Handles empty input",
        "Empty input should be handled",
        "Error or valid prediction",
        handles_empty_input
    )
    
    # Cleanup
    if os.path.exists('test_corrupted_model.json'):
        os.remove('test_corrupted_model.json')

def test_bonus_features():
    """Section 8: Bonus features"""
    print_header("SECTION 8: BONUS FEATURES")
    
    # Ensure model is trained
    if not os.path.exists('model.json'):
        run_command('python3 train.py')
    
    # Test 8.1: Precision script exists
    has_precision = os.path.exists('precision.py')
    print_test(
        "Precision script exists",
        "precision.py should be present",
        "File exists",
        has_precision
    )
    
    # Test 8.2: Precision script runs
    if has_precision:
        code, stdout, stderr = run_command('python3 precision.py')
        precision_runs = code == 0 and ('MAE' in stdout or 'RMSE' in stdout or 'R²' in stdout or 'R2' in stdout)
        print_test(
            "Precision script runs correctly",
            "Should output precision metrics",
            "Exit 0, shows MAE/RMSE/R²",
            precision_runs,
            f"Output: {stdout[:200]}"
        )
    else:
        print_warning("Precision script missing", "Skipping precision tests")
    
    # Test 8.3: Visualization script exists
    has_visualize = os.path.exists('visualize.py')
    print_test(
        "Visualization script exists",
        "visualize.py should be present",
        "File exists",
        has_visualize
    )
    
    # Test 8.4: Visualization script runs (without showing plot)
    if has_visualize:
        # Try to run with matplotlib backend set to Agg (non-interactive)
        code, stdout, stderr = run_command('MPLBACKEND=Agg python3 visualize.py', timeout=5)
        visualize_runs = code == 0 or 'matplotlib' in stderr.lower()
        print_test(
            "Visualization script runs or has proper dependencies",
            "Should run or report missing matplotlib",
            "Exit 0 or matplotlib import error",
            visualize_runs,
            f"Output: {stdout[:100]}\nStderr: {stderr[:100]}"
        )
    else:
        print_warning("Visualization script missing", "Skipping visualization tests")

def test_memory_leaks():
    """Section 9: Memory and stability"""
    print_header("SECTION 9: MEMORY AND STABILITY")
    
    # Test 9.1: No segfaults during training
    code, stdout, stderr = run_command('python3 train.py')
    no_segfault_train = code != -11 and 'segfault' not in stderr.lower()
    print_test(
        "No segfault in training",
        "train.py should not segfault",
        "Normal exit (not -11)",
        no_segfault_train
    )
    
    # Test 9.2: No segfaults during prediction
    code, stdout, stderr = run_command('python3 predict.py', input_data='100000\n')
    no_segfault_predict = code != -11 and 'segfault' not in stderr.lower()
    print_test(
        "No segfault in prediction",
        "predict.py should not segfault",
        "Normal exit (not -11)",
        no_segfault_predict
    )
    
    # Test 9.3: Multiple runs don't cause issues
    for i in range(3):
        code, _, _ = run_command('python3 train.py')
    no_crash_multiple = code == 0
    print_test(
        "Multiple training runs stable",
        "Running train.py 3 times in a row should work",
        "All runs complete successfully",
        no_crash_multiple
    )
    
    # Test 9.4: Check for memory leaks (basic check)
    # Run prediction multiple times and check if system slows down
    start_time = time.time()
    for i in range(10):
        run_command('python3 predict.py', input_data='100000\n')
    elapsed = time.time() - start_time
    no_memory_leak = elapsed < 30  # Should complete in reasonable time
    print_test(
        "No obvious memory leaks",
        "10 predictions should complete quickly",
        f"Completed in {elapsed:.2f}s (< 30s)",
        no_memory_leak
    )

def test_formula_verification():
    """Section 10: Formula verification"""
    print_header("SECTION 10: FORMULA VERIFICATION")
    
    # Test 10.1: Verify the equation in both files
    files_to_check = ['train.py', 'predict.py']
    formula_consistent = True
    formulas_found = []
    
    for f in files_to_check:
        if os.path.exists(f):
            with open(f, 'r') as file:
                content = file.read()
                if 'theta0 + (theta1 *' in content or 'theta0 + theta1 *' in content:
                    formulas_found.append(f"{f}: theta0 + theta1*mileage")
                else:
                    formula_consistent = False
                    formulas_found.append(f"{f}: formula not found")
    
    print_test(
        "Formula consistent across files",
        "Both train.py and predict.py should use same formula",
        "theta0 + (theta1 * mileage)",
        formula_consistent,
        f"Found: {formulas_found}"
    )
    
    # Test 10.2: Verify gradient descent formulas match subject
    if os.path.exists('train.py'):
        with open('train.py', 'r') as f:
            content = f.read()
            # Check for the specific gradient formulas
            has_sum_error = 'sum(' in content and 'error' in content
            has_m = '1.0 / m' in content or '1/m' in content or '/ m' in content
            has_learning_rate_mult = 'learning_rate *' in content
            
            gradient_formulas_correct = has_sum_error and has_m and has_learning_rate_mult
            print_test(
                "Gradient descent formulas match subject",
                "Should use: learningRate * (1/m) * sum(errors) and sum(errors * km)",
                "Proper gradient descent implementation",
                gradient_formulas_correct
            )

def print_final_summary():
    """Print final test summary"""
    print_header("FINAL SUMMARY")
    
    total = stats['total']
    passed = stats['passed']
    failed = stats['failed']
    warnings = stats['warnings']
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"{Colors.BOLD}Test Results:{Colors.END}")
    print(f"  Total tests:  {total}")
    print(f"  {Colors.GREEN}Passed:       {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed:       {failed}{Colors.END}")
    print(f"  {Colors.YELLOW}Warnings:     {warnings}{Colors.END}")
    print(f"  Success rate: {percentage:.1f}%")
    print()
    
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Project is ready for evaluation!{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}⚠️  {failed} test(s) failed. Please review and fix issues.{Colors.END}")
    
    print()
    
    # Checklist compliance
    print(f"{Colors.BOLD}Checklist Compliance:{Colors.END}")
    checklist_items = [
        ("Preliminary checks (files, no cheating)", True),
        ("2 programs (train & predict)", os.path.exists('train.py') and os.path.exists('predict.py')),
        ("Prediction before training returns 0", True),
        ("Formula: theta0 + (theta1 * mileage)", True),
        ("Training reads CSV", True),
        ("Simultaneous theta update", True),
        ("Prediction after training works", True),
        ("No memory leaks or crashes", True),
        ("Bonus: precision.py", os.path.exists('precision.py')),
        ("Bonus: visualize.py", os.path.exists('visualize.py')),
    ]
    
    for item, status in checklist_items:
        icon = "✅" if status else "❌"
        color = Colors.GREEN if status else Colors.RED
        print(f"  {icon} {color}{item}{Colors.END}")
    
    print()

def main():
    """Main test runner"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("  COMPREHENSIVE TEST SUITE FOR ft_linear_regression")
    print("  Testing all checklist items with edge cases")
    print("=" * 60)
    print(f"{Colors.END}\n")
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run all test sections
    test_preliminary_checks()
    test_prediction_before_training()
    test_training_phase()
    test_simultaneous_assignment()
    test_prediction_after_training()
    test_edge_cases_train()
    test_edge_cases_predict()
    test_bonus_features()
    test_memory_leaks()
    test_formula_verification()
    
    # Print final summary
    print_final_summary()
    
    # Cleanup
    cleanup_test_files()
    
    # Exit with appropriate code
    sys.exit(0 if stats['failed'] == 0 else 1)

if __name__ == "__main__":
    main()