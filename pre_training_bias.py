# pylint: skip-file
"""This module contains the implementation of the metrics for pre training bias evaluation"""
import pandas as pd
import numpy as np
from numpy import Infinity
from clarify_helper import pdfs_aligned_nonzero


class PreTrainingBias():
    """metrics implementation for pre training bias evaluation"""

    def _class_imbalance(self, n_a, n_d):
        return (n_a - n_d) / (n_a + n_d)

    def _difference_in_positive_proportions_of_labels(self, q_a, q_d):
        return q_a - q_d

    def _kl_divergence(self, p, q):
        kl_value = np.sum(p * np.log(self._divide(p, q)))
        if np.isnan(kl_value):
            return 0.0

        return kl_value

    def _divide(self, a, b) -> float:
        if b == 0 and a == 0:
            return 0.0
        if b == 0:
            if a < 0:
                return -Infinity
            return Infinity
        return a / b

    def class_imbalance(self, df, label, threshold=None):
        """returns the class imbalance for the given label"""
        facet_counts = df[label].value_counts(sort=True)
        if len(facet_counts) == 2:
            return self._class_imbalance(facet_counts.values[0], facet_counts.values[1])

        if threshold is None: # is not a binary attr
            raise ValueError("threshold not defined")
        a = len(df[df[label] > threshold])
        b = len(df[df[label] <= threshold])
        return self._class_imbalance(max(a, b), min(a, b))

    def class_imbalance_per_label(self, df, label, privileged_group) -> float:
        """returns the class imbalance for the given label and privileged group"""
        return self._class_imbalance((df[label].values == privileged_group).sum(),
                                    (df[label].values != privileged_group).sum())

    def kl_divergence(self, df, target, protected_attribute: str, privileged_group) -> float:
        """returns the kl divergence for the given target and protected attribute"""
        label = df[target]
        p_list = list()
        sensitive_facet_index = df[protected_attribute] != privileged_group
        unsensitive_facet_index = df[protected_attribute] == privileged_group
        p_list = pdfs_aligned_nonzero(
            label[unsensitive_facet_index], label[sensitive_facet_index])
        ks_val = 0
        for i, j in enumerate(p_list[0]):  # j = 0, 2 , i = 0
            ks_val += self._kl_divergence(j, p_list[1][i])
        return ks_val

    def ks(self, df, target, protected_attribute: str, privileged_group) -> float:
        """returns the ks for the given target and protected attribute"""
        label = df[target]
        p_list = list()
        sensitive_facet_index = df[protected_attribute] != privileged_group
        unsensitive_facet_index = df[protected_attribute] == privileged_group
        p_list = pdfs_aligned_nonzero(
            label[unsensitive_facet_index], label[sensitive_facet_index])
        ks_val = 0
        for i, j in enumerate(p_list[0]):
            ks_val = max(ks_val, abs(np.subtract(j, p_list[1][i])))
        return ks_val

    def cddl(self, df: pd.DataFrame, target: str, positive_outcome, protected_attribute,
                                                privileged_group, group_variable) -> float:
        """returns the cddl for the given target and protected attribute 
        grouping by the group_variable"""
        unique_groups = np.unique(df[group_variable])
        cdd = np.array([])
        counts = np.array([])
        for subgroup_variable in unique_groups:
            counts = np.append(
                counts, (df[group_variable].values == subgroup_variable).sum())
            num_a = len(df[(df[target] == positive_outcome) & (
                df[protected_attribute] != privileged_group) & (df[group_variable] == subgroup_variable)])
            denom_a = len(df[(df[target] == positive_outcome) &
                         (df[group_variable] == subgroup_variable)])
            a = num_a / denom_a if denom_a != 0 else 0
            num_d = len(df[(df[target] != positive_outcome) & (
                df[protected_attribute] != privileged_group) & (df[group_variable] == subgroup_variable)])
            denom_d = len(df[(df[target] != positive_outcome) &
                         (df[group_variable] == subgroup_variable)])
            d = num_d / denom_d if denom_d != 0 else 0
            cdd = np.append(cdd, d - a)
        return self._divide(np.sum(counts * cdd), np.sum(counts))

    def global_evaluation(self, df: pd.DataFrame, target: str, positive_outcome, 
                          protected_attribute, privileged_group, group_variable):
        """returns a dictionary with the metrics for the given target and protected attribute 
        grouping by the group_variable"""
        dic = {
            f"class imbalance ({protected_attribute})": 
                self.class_imbalance_per_label(df, protected_attribute, privileged_group),
            f"kl divergence ({protected_attribute})": 
                self.kl_divergence(df, target, protected_attribute, privileged_group),
            f"ks ({protected_attribute})": 
                self.ks(df, target, protected_attribute, privileged_group),
            f"cddl ({protected_attribute}, {group_variable})": 
                self.cddl(df, target, positive_outcome, protected_attribute, 
                          privileged_group, group_variable)
        }
        return dic

    def get_class_imbalance_permutation_values(self, df, label, n_repetitions, threshold=None):
        original_class_imbalance = self.class_imbalance(df, label, threshold)
        class_imbalance_permutation_values = []
        for i in range(n_repetitions):
            df_permuted = df.copy()
            df_permuted[label] = np.random.permutation(df[label])
            class_imbalance_permutation_values.append(self.class_imbalance(df_permuted, label, threshold))
        return class_imbalance_permutation_values, original_class_imbalance

    def get_ks_permutation_values(self, df, target, protected_attribute, privileged_group, n_repetitions):
        original_ks = self.ks(df, target, protected_attribute, privileged_group)
        ks_permutation_values = []
        for i in range(n_repetitions):
            df_permuted = df.copy()
            df_permuted[target] = np.random.permutation(df[target])
            ks_permutation_values.append(self.ks(df_permuted, target, protected_attribute, privileged_group))
        return ks_permutation_values, original_ks

    def get_cddl_permutation_values(self, df, target, positive_outcome, protected_attribute, privileged_group, group_variable, n_repetitions):
        original_cddl = self.cddl(df, target, positive_outcome, protected_attribute, privileged_group, group_variable)
        cddl_permutation_values = []
        for i in range(n_repetitions):
            df_permuted = df.copy()
            df_permuted[target] = np.random.permutation(df[target])
            cddl_permutation_values.append(self.cddl(df_permuted, target, positive_outcome, protected_attribute, privileged_group, group_variable))
        return cddl_permutation_values, original_cddl

    def get_kl_divergence_permutation_values(self, df, target, protected_attribute, privileged_group, n_repetitions):
        original_kl_divergence = self.kl_divergence(df, target, protected_attribute, privileged_group)
        kl_divergence_permutation_values = []
        for i in range(n_repetitions):
            df_permuted = df.copy()
            df_permuted[target] = np.random.permutation(df[target])
            kl_divergence_permutation_values.append(self.kl_divergence(df_permuted, target, protected_attribute, privileged_group))
        return kl_divergence_permutation_values, original_kl_divergence
