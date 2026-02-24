#!/usr/bin/env python
"""
Aggregate parameter recovery results from individual simulations.

This script reads all individual simulation results (stored as JSON files)
and combines them into a single, easy-to-use format for plotting and analysis.

Usage:
    python aggregate_results.py --input-dir ./recovery_results/ --output-file recovery_results.pkl

Output format:
    - Pickle file containing a dictionary with all results
    - CSV files for easy inspection in Excel/other tools
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def aggregate_results(input_dir: Path) -> Dict[str, Any]:
    """
    Aggregate all individual simulation results into a single structure.
    
    Returns:
        Dictionary with structure:
        {
            'simulations': [list of individual simulation results],
            'summary_df': DataFrame with all results for easy analysis,
            'metadata': processing metadata,
        }
    """
    
    # Find all simulation result files
    json_files = sorted(input_dir.glob('sim_*.json'))
    
    if not json_files:
        print(f"Error: No simulation result files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(json_files)} simulation result files")
    
    # Load all results
    all_results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(all_results)} simulations")
    
    # Create comprehensive DataFrame
    rows = []
    for result in all_results:
        row = {
            'sim_id': result['sim_id'],
        }
        
        # Add true parameters
        for param, value in result['true_params'].items():
            row[f'true_{param}'] = value
        
        # Add TADA posterior means and HDI
        for param, value in result['tada_posterior'].items():
            row[f'tada_{param}'] = value
        
        # Add aDDM posterior means and HDI
        for param, value in result['addm_posterior'].items():
            row[f'addm_{param}'] = value
        
        # Add timing information
        for task, duration in result['timing'].items():
            row[f'time_{task}'] = duration
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Calculate recovery errors for easier plotting
    params_to_recover = ['eta', 'kappa', 'a', 'x0']
    
    for param in params_to_recover:
        true_col = f'true_{param}'
        
        if f'tada_{param}' in summary_df.columns:
            summary_df[f'tada_{param}_error'] = summary_df[f'tada_{param}'] - summary_df[true_col]
            summary_df[f'tada_{param}_rel_error'] = (summary_df[f'tada_{param}_error'] / 
                                                     summary_df[true_col].replace(0, np.nan))
        
        if f'addm_{param}' in summary_df.columns:
            summary_df[f'addm_{param}_error'] = summary_df[f'addm_{param}'] - summary_df[true_col]
            summary_df[f'addm_{param}_rel_error'] = (summary_df[f'addm_{param}_error'] / 
                                                     summary_df[true_col].replace(0, np.nan))
    
    # Create aggregated result dictionary
    aggregated = {
        'simulations': all_results,
        'summary_df': summary_df,
        'metadata': {
            'num_simulations': len(all_results),
            'parameters_recovered': params_to_recover,
            'created_from': str(input_dir),
        }
    }
    
    return aggregated


def print_summary_statistics(results: Dict[str, Any]) -> None:
    """Print summary statistics of the recovery results."""
    
    df = results['summary_df']
    params = results['metadata']['parameters_recovered']
    
    print("\n" + "="*70)
    print("PARAMETER RECOVERY SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal simulations: {len(df)}")
    
    for param in params:
        print(f"\n{param.upper()} recovery:")
        print(f"  True parameters - Mean: {df[f'true_{param}'].mean():.4f}, "
              f"Std: {df[f'true_{param}'].std():.4f}")
        
        if f'tada_{param}' in df.columns:
            print(f"  TADA recovered  - Mean: {df[f'tada_{param}'].mean():.4f}, "
                  f"Std: {df[f'tada_{param}'].std():.4f}")
            if f'tada_{param}_error' in df.columns:
                error = df[f'tada_{param}_error'].dropna()
                print(f"  TADA error      - Mean: {error.mean():.4f}, "
                      f"Std: {error.std():.4f}, RMSE: {np.sqrt((error**2).mean()):.4f}")
        
        if f'addm_{param}' in df.columns:
            print(f"  aDDM recovered  - Mean: {df[f'addm_{param}'].mean():.4f}, "
                  f"Std: {df[f'addm_{param}'].std():.4f}")
            if f'addm_{param}_error' in df.columns:
                error = df[f'addm_{param}_error'].dropna()
                print(f"  aDDM error      - Mean: {error.mean():.4f}, "
                      f"Std: {error.std():.4f}, RMSE: {np.sqrt((error**2).mean()):.4f}")


def save_results(results: Dict[str, Any], output_file: Path, csv_dir: Path = None) -> None:
    """Save aggregated results to pickle and CSV formats."""
    
    # Save as pickle
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to pickle: {output_file}")
    
    # Save summary DataFrame as CSV
    if csv_dir is None:
        csv_dir = output_file.parent
    
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Full results CSV
    full_csv = csv_dir / output_file.stem / "_full.csv"
    full_csv.parent.mkdir(parents=True, exist_ok=True)
    results['summary_df'].to_csv(full_csv, index=False)
    print(f"Full results saved to CSV: {full_csv}")
    
    # Compact results CSV (just sim_id, true params, posterior means, and HDI)
    params = results['metadata']['parameters_recovered']
    compact_cols = ['sim_id']
    for param in params:
        compact_cols.append(f'true_{param}')
        if f'tada_{param}' in results['summary_df'].columns:
            compact_cols.append(f'tada_{param}')
            # Add HDI columns if they exist
            if f'tada_{param}_hdi_3%' in results['summary_df'].columns:
                compact_cols.append(f'tada_{param}_hdi_3%')
            if f'tada_{param}_hdi_97%' in results['summary_df'].columns:
                compact_cols.append(f'tada_{param}_hdi_97%')
        if f'addm_{param}' in results['summary_df'].columns:
            compact_cols.append(f'addm_{param}')
            # Add HDI columns if they exist
            if f'addm_{param}_hdi_3%' in results['summary_df'].columns:
                compact_cols.append(f'addm_{param}_hdi_3%')
            if f'addm_{param}_hdi_97%' in results['summary_df'].columns:
                compact_cols.append(f'addm_{param}_hdi_97%')
    
    # Filter to only existing columns
    compact_cols = [col for col in compact_cols if col in results['summary_df'].columns]
    compact_df = results['summary_df'][compact_cols]
    compact_csv = csv_dir / (output_file.stem + "_compact.csv")
    compact_df.to_csv(compact_csv, index=False)
    print(f"Compact results saved to CSV: {compact_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate parameter recovery results from multiple simulations"
    )
    parser.add_argument('--input-dir', type=str, default='./recovery_results/',
                       help='Input directory containing individual simulation JSON files')
    parser.add_argument('--output-file', type=str, default='recovery_results.pkl',
                       help='Output pickle file')
    parser.add_argument('--csv-dir', type=str, default=None,
                       help='Directory for CSV exports (defaults to same as output)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    
    print(f"Aggregating results from: {input_dir}")
    
    # Aggregate results
    results = aggregate_results(input_dir)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Save results
    save_results(results, output_file, csv_dir)
    
    print("\n" + "="*70)
    print("Aggregation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
