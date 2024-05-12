import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from TreeModelsFromScratch.RandomForest import RandomForest
from utils import simulate_data_strobl
import os

def calculate_auc_roc(y_true, y_scores):
    """Calculate the AUC-ROC score."""
    return roc_auc_score(y_true, y_scores)

def evaluate_model_for_k(k, iterations, relevance, n_trees, depth_dof):
    roc_aucs, mdi_importances, shap_values_all = [], [], []
    print(f"Starting evaluation for k={k}")
    for _ in range(iterations):
        X, y = simulate_data_strobl(n=300, relevance=relevance)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = RandomForest(treetype="classification", n_trees=n_trees, k=k, oob_SHAP=True, depth_dof=depth_dof)
        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]
        roc_aucs.append(calculate_auc_roc(y_test, y_scores))
        mdi_importances.append(model.feature_importances_)
        shap_values_all.append(np.mean(np.abs(model.oob_SHAP_values), axis=0))
    print(f"Done with k={k}")
    return {
        'k': k,
        'roc_auc': np.mean(roc_aucs),
        'mdi_importances': np.mean(mdi_importances, axis=0),
        'shap_values': np.mean(shap_values_all, axis=0)
    }

def plot_results(all_results, depth_dof):
    output_folder = 'outputs_runSHAP'
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    for relevance_result in all_results:
        relevance = relevance_result['relevance']
        plt.figure(figsize=(18, 6))

        ks = [result['k'] for result in relevance_result['results']]
        rocs = [result['roc_auc'] for result in relevance_result['results']]
        mdi_importances = [result['mdi_importances'] for result in relevance_result['results']]
        shap_importances = [result['shap_values'] for result in relevance_result['results']]

        plt.subplot(1, 3, 1)
        plt.plot(ks, rocs, label='ROC AUC', marker="o")
        plt.title(f'ROC AUC vs k - Relevance {relevance:.2f}')
        plt.xlabel('k')
        plt.ylabel('ROC AUC')
        plt.legend()

        plt.subplot(1, 3, 2)
        for feature_idx in range(len(mdi_importances[0])):
            mdi_feature_means = [mdi[feature_idx] for mdi in mdi_importances]
            plt.plot(ks, mdi_feature_means, label=f'X_{feature_idx + 1}', marker="o")
        plt.title(f'MDI Importance vs k - Relevance {relevance:.2f}')
        plt.xlabel('k')
        plt.ylabel('MDI Importance')
        plt.legend()

        plt.subplot(1, 3, 3)
        for feature_idx in range(len(shap_importances[0])):
            shap_feature_means = [shap[feature_idx] for shap in shap_importances]
            plt.plot(ks, shap_feature_means, label=f'X_{feature_idx + 1}', marker="o")
        plt.title(f'SHAP Importance vs k - Relevance {relevance:.2f}')
        plt.xlabel('k')
        plt.ylabel('SHAP Importance')
        plt.legend()

        plt.tight_layout()
        
        # Save the plot to the specified output directory
        filename = f"{output_folder}/results_depth_dof_{depth_dof}_relevance_{relevance:.2f}.png"
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory
        print(f"Saved plot as {filename}")

def main(n_trees, n_cores, iterations, relevance_values, depth_dof):
    k_values = range(1, 31)
    all_results = []
    for relevance in relevance_values:
        print(f"Evaluating models for relevance={relevance}")
        results = Parallel(n_jobs=n_cores)(
            delayed(evaluate_model_for_k)(k, iterations, relevance, n_trees, depth_dof) for k in k_values
        )
        all_results.append({'relevance': relevance, 'results': results})

    # Save results to a pickle file
    with open('evaluation_results.pkl', 'wb') as file:
        pickle.dump(all_results, file)

    # Plotting and saving the results
    plot_results(all_results, depth_dof)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RandomForest models with varying k and plot results.")
    parser.add_argument('--n_trees', type=int, default=25, help='Number of trees in the forest')
    parser.add_argument('--n_cores', type=int, default=1, help='Number of cores to use for parallel processing')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations per k-value')
    parser.add_argument('--depth_dof', action='store_true', help='Enable depth degrees of freedom (default: False)')
    
    args = parser.parse_args()

    relevance_values = [0, 0.05, 0.1, 0.15, 0.2]  # Adjust as needed
    main(args.n_trees, args.n_cores, args.iterations, relevance_values, args.depth_dof)
