#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

from helper_code import load_patient_data, get_pitch, load_challenge_outputs, compare_strings

def find_challenge_files(label_folder, output_folder):
    label_files, output_files = [], []
    for label_file in sorted(os.listdir(label_folder)):
        label_file_path = os.path.join(label_folder, label_file)
        if os.path.isfile(label_file_path) and label_file.lower().endswith('.txt'):
            root, _ = os.path.splitext(label_file)
            output_file_path = os.path.join(output_folder, root + '.csv')
            if os.path.isfile(output_file_path):
                label_files.append(label_file_path)
                output_files.append(output_file_path)
            else:
                print(f" Warning: Missing output file for {label_file}")
    return label_files, output_files

def load_pitchs(label_files):
    valid_indices, labels = [], []
    included_labels = ["Low", "Medium", "High"]
    for i, file in enumerate(label_files):
        data = load_patient_data(file)
        label = get_pitch(data)
        if label in included_labels:
            labels.append([
                int(label == "Low"),
                int(label == "Medium"),
                int(label == "High")
            ])
            valid_indices.append(i)
    return np.array(labels, dtype=int), valid_indices

def load_classifier_outputs(output_files, valid_indices):
    binary_outputs, scalar_outputs = [], []
    included_labels = ["Low", "Medium", "High"]
    filtered_output_files = [output_files[i] for i in valid_indices]
    for file in filtered_output_files:
        _, patient_classes, _, patient_scalar_outputs = load_challenge_outputs(file)
        binary_output = [0, 0, 0]
        scalar_output = [0.0, 0.0, 0.0]
        for j, x in enumerate(included_labels):
            for k, y in enumerate(patient_classes):
                if compare_strings(x, y):
                    scalar_output[j] = patient_scalar_outputs[k]
                    binary_output[j] = int(patient_scalar_outputs[k] >= 0.5)
        binary_outputs.append(binary_output)
        scalar_outputs.append(scalar_output)
    return np.array(binary_outputs, dtype=int), np.array(scalar_outputs, dtype=np.float64)

def compute_auc(labels, outputs):
    try:
        aucs = [roc_auc_score(labels[:, i], outputs[:, i]) for i in range(3)]
        prcs = [average_precision_score(labels[:, i], outputs[:, i]) for i in range(3)]
    except ValueError:
        aucs = [0.5, 0.5, 0.5]
        prcs = [0.5, 0.5, 0.5]
    return aucs, prcs

def compute_f_measure(labels, outputs):
    scores = [f1_score(labels[:, i], outputs[:, i]) for i in range(3)]
    return np.mean(scores), scores

def compute_accuracy(labels, outputs):
    scores = [accuracy_score(labels[:, i], outputs[:, i]) for i in range(3)]
    return np.mean(scores), scores

def compute_weighted_accuracy(labels, outputs):
    weights = np.array([[2, 1, 1], [1, 3, 1], [1, 1, 3]])
    confusion = np.zeros((3, 3))
    for i in range(len(labels)):
        true_class = np.argmax(labels[i])
        pred_class = np.argmax(outputs[i])
        confusion[pred_class, true_class] += 1
    return np.trace(weights * confusion) / np.sum(weights * confusion)

#  Visualizations
def generate_visualizations_multiclass(true_onehot, predicted_probs, class_names=["Low", "Medium", "High"], output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.argmax(true_onehot, axis=1)
    y_pred = np.argmax(predicted_probs, axis=1)

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_onehot.ravel(), predicted_probs.ravel())
    plt.figure()
    plt.plot(fpr, tpr, label="Overall ROC")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall ROC Curve")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_roc.png"))
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(true_onehot.ravel(), predicted_probs.ravel())
    plt.figure()
    plt.plot(recall, precision, label="Overall PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_pr.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Multiclass')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, "overall_confusion_matrix_multiclass.png"))
    plt.close()

#  Main evaluation function
def evaluate_model(label_folder, output_folder):
    print("🔍 Evaluating model...")
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    pitch_labels, valid_indices = load_pitchs(label_files)
    pitch_binary_outputs, pitch_scalar_outputs = load_classifier_outputs(output_files, valid_indices)

    threshold = 0.5
    pitch_binary_outputs = (pitch_scalar_outputs >= threshold).astype(int)

    class_names = ["Low", "Medium", "High"]
    generate_visualizations_multiclass(pitch_labels, pitch_scalar_outputs, class_names)

    aurocs, auprcs = compute_auc(pitch_labels, pitch_scalar_outputs)
    f_measure, f_classes = compute_f_measure(pitch_labels, pitch_binary_outputs)
    accuracy, acc_classes = compute_accuracy(pitch_labels, pitch_binary_outputs)
    weighted_acc = compute_weighted_accuracy(pitch_labels, pitch_binary_outputs)

    return class_names, aurocs, auprcs, f_measure, f_classes, accuracy, acc_classes, weighted_acc

#  Save scores
def print_and_save_scores(filename, pitch_scores):
    classes, auroc, auprc, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy = pitch_scores
    output_string = f"""
#Pitch scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy
{np.mean(auroc):.3f},{np.mean(auprc):.3f},{f_measure:.3f},{accuracy:.3f},{weighted_accuracy:.3f}

#Pitch scores (per class)
Classes,Low,Medium,High
AUROC,{auroc[0]:.3f},{auroc[1]:.3f},{auroc[2]:.3f}
AUPRC,{auprc[0]:.3f},{auprc[1]:.3f},{auprc[2]:.3f}
F-measure,{f_measure_classes[0]:.3f},{f_measure_classes[1]:.3f},{f_measure_classes[2]:.3f}
Accuracy,{accuracy_classes[0]:.3f},{accuracy_classes[1]:.3f},{accuracy_classes[2]:.3f}
"""
    print(output_string)
    with open(filename, 'w') as f:
        f.write(output_string.strip())
    print(f"✅ Scores saved to {filename}")

#  Entry point
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python evaluate_model_pitch.py <label_folder> <output_folder> <scores.csv>")
        sys.exit(1)

    pitch_scores = evaluate_model(sys.argv[1], sys.argv[2])
    print_and_save_scores(sys.argv[3], pitch_scores)
    print("Model Evaluation Completed. Check scores.csv and plots/ folder for visualizations.")
