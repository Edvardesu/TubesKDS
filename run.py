#!/usr/bin/env python3
"""
Script utama untuk menjalankan simulasi Spatial Prisoner's Dilemma
Mendukung mode visualisasi dan mode batch untuk eksperimen
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from model import SpatialPDModel
from agent import Strategy
import json
import time
from datetime import datetime
import sys
import traceback

def run_visualization():
    """
    Menjalankan simulasi dengan visualisasi web Mesa
    """
    print("🚀 Menjalankan visualisasi web...")
    print("🌐 Buka browser dan kunjungi: http://localhost:8521")
    print("⏹️ Tekan Ctrl+C untuk menghentikan server")
    
    try:
        from server import server
        server.launch()
    except ImportError as e:
        print(f"❌ Error importing server: {e}")
        print("Pastikan file server.py ada dan dapat diimport")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        traceback.print_exc()
        sys.exit(1)

def run_batch_simulation(config):
    """
    Menjalankan simulasi batch tanpa visualisasi
    
    Args:
        config: Dictionary konfigurasi simulasi
    """
    print(f"📊 Menjalankan simulasi batch dengan konfigurasi:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Buat model
        model = SpatialPDModel(
            width=config.get('width', 100),
            height=config.get('height', 100),
            density=config.get('density', 0.8),
            neighborhood_type=config.get('neighborhood_type', 'moore'),
            update_type=config.get('update_type', 'synchronous'),
            initial_cooperation_rate=config.get('initial_cooperation_rate', 0.5),
            mutation_rate=config.get('mutation_rate', 0.01),
            payoff_matrix=config.get('payoff_matrix', None)
        )
        
        # Jalankan simulasi
        steps = config.get('steps', 100)
        print(f"🏃 Menjalankan {steps} steps...")
        
        start_time = time.time()
        for i in range(steps):
            model.step()
            if (i + 1) % 10 == 0:
                cooperation_rate = model.get_cooperation_rate()
                avg_score = model.get_average_score()
                print(f"Step {i+1}: Cooperation Rate = {cooperation_rate:.3f}, Avg Score = {avg_score:.2f}")
        
        end_time = time.time()
        print(f"✅ Simulasi selesai dalam {end_time - start_time:.2f} detik")
        
        return model
        
    except Exception as e:
        print(f"❌ Error in batch simulation: {e}")
        traceback.print_exc()
        sys.exit(1)

def analyze_results(model, save_plots=True, output_dir="results"):
    """
    Analisis dan visualisasi hasil simulasi
    
    Args:
        model: Model yang telah dijalankan
        save_plots: Apakah menyimpan plot
        output_dir: Directory untuk menyimpan hasil
    """
    try:
        import os
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Ambil data
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()
        
        # Setup plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spatial Prisoner\'s Dilemma - Analysis Results', fontsize=16)
        
        # 1. Cooperation Rate over Time
        axes[0, 0].plot(model_data.index, model_data['Cooperation_Rate'], 'b-', linewidth=2)
        axes[0, 0].set_title('Cooperation Rate Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Cooperation Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Average Score over Time
        axes[0, 1].plot(model_data.index, model_data['Average_Score'], 'g-', linewidth=2)
        axes[0, 1].set_title('Average Score Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Average Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Population Dynamics
        axes[0, 2].plot(model_data.index, model_data['Cooperators'], 'b-', label='Cooperators', linewidth=2)
        axes[0, 2].plot(model_data.index, model_data['Defectors'], 'r-', label='Defectors', linewidth=2)
        axes[0, 2].set_title('Population Dynamics')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Number of Agents')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Cooperation Clustering
        axes[1, 0].plot(model_data.index, model_data['Clustering_Cooperators'], 'purple', linewidth=2)
        axes[1, 0].set_title('Cooperation Clustering')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Clustering Coefficient')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # 5. Score Distribution (final step)
        if len(agent_data) > 0:
            final_step = agent_data.index.get_level_values('Step').max()
            final_data = agent_data.xs(final_step, level='Step')
            
            axes[1, 1].hist(final_data['Score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].set_title('Score Distribution (Final Step)')
            axes[1, 1].set_xlabel('Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Spatial Distribution (final step)
        spatial_data = model.get_spatial_distribution()
        im = axes[1, 2].imshow(spatial_data, cmap='RdBu', vmin=0, vmax=1)
        axes[1, 2].set_title('Spatial Distribution (Blue=Coop, Red=Defect)')
        axes[1, 2].set_xlabel('X')
        axes[1, 2].set_ylabel('Y')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{output_dir}/analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot disimpan di: {plot_filename}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("📈 SUMMARY STATISTICS")
        print("="*50)
        
        final_coop_rate = model_data['Cooperation_Rate'].iloc[-1]
        final_avg_score = model_data['Average_Score'].iloc[-1]
        final_clustering = model_data['Clustering_Cooperators'].iloc[-1]
        
        print(f"Final Cooperation Rate: {final_coop_rate:.3f}")
        print(f"Final Average Score: {final_avg_score:.2f}")
        print(f"Final Clustering: {final_clustering:.3f}")
        print(f"Score Variance: {model_data['Score_Variance'].iloc[-1]:.2f}")
        
        # Stability analysis
        last_10_steps = model_data.tail(10)
        coop_rate_std = last_10_steps['Cooperation_Rate'].std()
        print(f"Cooperation Rate Stability (last 10 steps std): {coop_rate_std:.4f}")
        
        if coop_rate_std < 0.01:
            print("Status: ✅ CONVERGED (stable)")
        elif coop_rate_std < 0.05:
            print("Status: ⚠️ QUASI-STABLE")
        else:
            print("Status: 🔄 STILL EVOLVING")
        
        print("="*50)
        
        return model_data, agent_data
        
    except Exception as e:
        print(f"❌ Error in analyze_results: {e}")
        traceback.print_exc()

def compare_scenarios():
    """
    Membandingkan berbagai skenario simulasi
    """
    print("🔍 Menjalankan perbandingan skenario...")
    
    scenarios = [
        {
            "name": "Classic PD",
            "payoff_matrix": {"CC": [3, 3], "CD": [0, 5], "DC": [5, 0], "DD": [1, 1]},
            "neighborhood_type": "moore",
            "update_type": "synchronous"
        },
        {
            "name": "Hawk-Dove",
            "payoff_matrix": {"CC": [3, 3], "CD": [1, 4], "DC": [4, 1], "DD": [0, 0]},
            "neighborhood_type": "moore", 
            "update_type": "synchronous"
        },
        {
            "name": "Stag Hunt",
            "payoff_matrix": {"CC": [4, 4], "CD": [0, 3], "DC": [3, 0], "DD": [2, 2]},
            "neighborhood_type": "moore",
            "update_type": "synchronous"
        },
        {
            "name": "Von Neumann vs Moore",
            "payoff_matrix": {"CC": [3, 3], "CD": [0, 5], "DC": [5, 0], "DD": [1, 1]},
            "neighborhood_type": "von_neumann",
            "update_type": "synchronous"
        }
    ]
    
    results = {}
    
    try:
        for scenario in scenarios:
            print(f"\n🎯 Menjalankan skenario: {scenario['name']}")
            
            config = {
                "width": 50,
                "height": 50,
                "density": 0.8,
                "steps": 100,
                "initial_cooperation_rate": 0.5,
                "mutation_rate": 0.01,
                **scenario
            }
            
            model = run_batch_simulation(config)
            model_data = model.datacollector.get_model_vars_dataframe()
            
            results[scenario['name']] = {
                "final_cooperation": model_data['Cooperation_Rate'].iloc[-1],
                "final_score": model_data['Average_Score'].iloc[-1],
                "data": model_data
            }
        
        # Plot perbandingan
        plt.figure(figsize=(15, 5))
        
        # Cooperation Rate comparison
        plt.subplot(1, 3, 1)
        for name, result in results.items():
            plt.plot(result['data'].index, result['data']['Cooperation_Rate'], 
                    label=name, linewidth=2)
        plt.title('Cooperation Rate Comparison')
        plt.xlabel('Step')
        plt.ylabel('Cooperation Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Average Score comparison
        plt.subplot(1, 3, 2)
        for name, result in results.items():
            plt.plot(result['data'].index, result['data']['Average_Score'], 
                    label=name, linewidth=2)
        plt.title('Average Score Comparison')
        plt.xlabel('Step')
        plt.ylabel('Average Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final values bar chart
        plt.subplot(1, 3, 3)
        names = list(results.keys())
        final_coops = [results[name]['final_cooperation'] for name in names]
        
        bars = plt.bar(range(len(names)), final_coops, alpha=0.7)
        plt.title('Final Cooperation Rates')
        plt.ylabel('Cooperation Rate')
        plt.xticks(range(len(names)), [name.replace(' ', '\n') for name in names], rotation=45)
        
        # Color bars
        for i, bar in enumerate(bars):
            if final_coops[i] > 0.5:
                bar.set_color('blue')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison summary
        print("\n" + "="*60)
        print("🏆 SCENARIO COMPARISON SUMMARY")
        print("="*60)
        for name, result in results.items():
            print(f"{name:20} | Coop: {result['final_cooperation']:.3f} | Score: {result['final_score']:.2f}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error in compare_scenarios: {e}")
        traceback.print_exc()

def main():
    """
    Fungsi utama dengan argument parsing
    """
    parser = argparse.ArgumentParser(description='Spatial Prisoner\'s Dilemma Simulator')
    parser.add_argument('--mode', choices=['visual', 'batch', 'compare'], 
                       default='visual', help='Mode simulasi')
    parser.add_argument('--config', help='File konfigurasi JSON untuk mode batch')
    parser.add_argument('--steps', type=int, default=100, help='Jumlah steps untuk simulasi')
    parser.add_argument('--width', type=int, default=50, help='Lebar grid')
    parser.add_argument('--height', type=int, default=50, help='Tinggi grid')
    parser.add_argument('--neighborhood', choices=['moore', 'von_neumann'], 
                       default='moore', help='Tipe neighborhood')
    parser.add_argument('--update', choices=['synchronous', 'asynchronous'],
                       default='synchronous', help='Tipe update')
    parser.add_argument('--output', default='results', help='Directory output')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'visual':
            run_visualization()
            
        elif args.mode == 'batch':
            if args.config:
                # Load dari file JSON
                try:
                    with open(args.config, 'r') as f:
                        config = json.load(f)
                except FileNotFoundError:
                    print(f"❌ File konfigurasi tidak ditemukan: {args.config}")
                    sys.exit(1)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing JSON: {e}")
                    sys.exit(1)
            else:
                # Gunakan parameter command line
                config = {
                    'width': args.width,
                    'height': args.height,
                    'steps': args.steps,
                    'neighborhood_type': args.neighborhood,
                    'update_type': args.update,
                    'density': 0.8,
                    'initial_cooperation_rate': 0.5,
                    'mutation_rate': 0.01
                }
            
            model = run_batch_simulation(config)
            analyze_results(model, save_plots=True, output_dir=args.output)
            
            # Simpan hasil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{args.output}/simulation_{timestamp}"
            model.save_results(filename)
            
        elif args.mode == 'compare':
            compare_scenarios()
            
    except KeyboardInterrupt:
        print("\n⏹️ Simulasi dihentikan oleh user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()