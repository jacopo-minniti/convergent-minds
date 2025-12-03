import xarray as xr

from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore import load_dataset, load_metric
from brainscore.artificial_subject import ArtificialSubject
from brainscore.data.pereira2018 import BIBTEX
from brainscore.utils.ceiling import ceiling_normalize
from brainscore.utils.s3 import load_from_s3


def Pereira2018_243sentences():
    return _Pereira2018ExperimentLinear(experiment='243sentences', ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))


def Pereira2018_384sentences():
    return _Pereira2018ExperimentLinear(experiment='384sentences', ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
    ))


def Pereira2018_243sentences_cka():
    return _Pereira2018Experiment(experiment='243sentences', metric='cka')


def Pereira2018_384sentences_cka():
    return _Pereira2018Experiment(experiment='384sentences', metric='cka')


class _Pereira2018ExperimentLinear(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the Pereira2018 benchmark introduced
    in Schrimpf et al. 2021 (https://www.pnas.org/doi/10.1073/pnas.2105646118), but:

    * computes neural alignment to each of the two experiments ({243,384}sentences) separately, as well as ceilings
    * requires the model to have committed to neural readouts (e.g. layer 41 corresponds to the language system),
        rather than testing every layer separately

    Each of these benchmarks evaluates one of the two experiments, the overall Pereira2018-linear score is the mean of
    the two ceiling-normalized scores.
    """

    def __init__(self, experiment: str, ceiling_s3_kwargs: dict):
        self.data = self._load_data(experiment)
        self.metric = load_metric('linear_pearsonr')
        identifier = f'Pereira2018.{experiment}-linear'
        ceiling = self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_Pereira2018ExperimentLinear, self).__init__(
            identifier=identifier,
            version=1,
            parent='Pereira2018-linear',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self, experiment: str) -> NeuroidAssembly:
        data = load_dataset('Pereira2018.language')
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data.attrs['identifier'] = f"{data.identifier}.{experiment}"
        return data

    def _load_ceiling(self, identifier: str, version_id: str, sha1: str, assembly_prefix="ceiling_", raw_kwargs=None):
        ceiling = load_from_s3(identifier, cls=Score, assembly_prefix=assembly_prefix, version_id=version_id, sha1=sha1)
        if raw_kwargs:  # recursively load raw attributes
            raw = self._load_ceiling(identifier=identifier, assembly_prefix=assembly_prefix + "raw_", **raw_kwargs)
            ceiling.attrs['raw'] = raw
        return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        passages = self.data['passage_label'].values
        predictions = []
        for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
            passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
            passage_stimuli = stimuli[passage_indexer]
            passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
            passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
            predictions.append(passage_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        raw_score = self.metric(predictions, self.data)
        score = ceiling_normalize(raw_score, self.ceiling)
        return score


class _Pereira2018Experiment(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018, using various metrics (CKA, RDM, etc.).
    """

    def __init__(self, experiment: str, metric: str):
        self.data = self._load_data(experiment)
        self.metric = load_metric(metric)
        identifier = f'Pereira2018.{experiment}-{metric}'
        super(_Pereira2018Experiment, self).__init__(
            identifier=identifier,
            version=1,
            parent=f'Pereira2018-{metric}',
            ceiling=None,
            bibtex=BIBTEX)

    def _load_data(self, experiment: str) -> NeuroidAssembly:
        data = load_dataset('Pereira2018.language')
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data.attrs['identifier'] = f"{data.identifier}.{experiment}"
        return data

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        passages = self.data['passage_label'].values
        predictions = []
        for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
            passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
            passage_stimuli = stimuli[passage_indexer]
            passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
            passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
            predictions.append(passage_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        raw_score = self.metric(predictions, self.data)
        return raw_score


def Pereira2018_243sentences_partialr2():
    return _Pereira2018ExperimentPartialR2(experiment='243sentences', ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))


def Pereira2018_384sentences_partialr2():
    return _Pereira2018ExperimentPartialR2(experiment='384sentences', ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
    ))


class _Pereira2018ExperimentPartialR2(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment is evaluated via Partial R2 with objective features.
    """

    def __init__(self, experiment: str, ceiling_s3_kwargs: dict = None, topic_wise_cv: bool = True):
        self.data = self._load_data(experiment)
        from alignment.metrics.linear_partial_r2 import linear_partial_r2
        self.metric = linear_partial_r2
        identifier = f'Pereira2018.{experiment}-partialr2'
        self.experiment = experiment
        self.topic_wise_cv = topic_wise_cv
        
        ceiling = None
        if ceiling_s3_kwargs:
            # Load ceiling using the linear benchmark identifier
            ceiling_identifier = f'Pereira2018.{experiment}-linear'
            ceiling = self._load_ceiling(identifier=ceiling_identifier, **ceiling_s3_kwargs)
            
        super(_Pereira2018ExperimentPartialR2, self).__init__(
            identifier=identifier,
            version=1,
            parent=f'Pereira2018-partialr2',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self, experiment: str) -> NeuroidAssembly:
        data = load_dataset('Pereira2018.language')
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data.attrs['identifier'] = f"{data.identifier}.{experiment}"
        return data

    def _load_ceiling(self, identifier: str, version_id: str, sha1: str, assembly_prefix="ceiling_", raw_kwargs=None):
        ceiling = load_from_s3(identifier, cls=Score, assembly_prefix=assembly_prefix, version_id=version_id, sha1=sha1)
        if raw_kwargs:  # recursively load raw attributes
            raw = self._load_ceiling(identifier=identifier, assembly_prefix=assembly_prefix + "raw_", **raw_kwargs)
            ceiling.attrs['raw'] = raw
        return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        passages = self.data['passage_label'].values
        predictions = []
        for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
            passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
            passage_stimuli = stimuli[passage_indexer]
            passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
            passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
            predictions.append(passage_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        
        # Load X_obj
        import os
        import numpy as np
        # Assuming data is in 'data/' relative to CWD
        # Filename convention from precompute script: pereira2018_{experiment}_obj.npz
        # e.g. pereira2018_243_obj.npz
        filename = f"pereira2018_{self.experiment.replace('sentences', '')}_obj.npz"
        filepath = os.path.join("data", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Objective features file not found at {filepath}. Please run precompute script.")
            
        data_obj = np.load(filepath, allow_pickle=True)
        X_obj = data_obj['X_obj']
        obj_stimulus_ids = data_obj['stimulus_ids']
        
        # Align X_obj to predictions (X_llm)
        # predictions has stimulus_id coordinate
        pred_stimulus_ids = predictions['stimulus_id'].values
        
        # Create a mapping from stimulus_id to index in X_obj
        obj_id_to_idx = {sid: i for i, sid in enumerate(obj_stimulus_ids)}
        
        # Reorder X_obj
        indices = [obj_id_to_idx[sid] for sid in pred_stimulus_ids]
        X_obj_aligned = X_obj[indices]
        
        # Prepare X_llm and y
        X_llm = predictions.values
        
        # Align y (self.data) to predictions
        # Use manual alignment via stimulus_id to avoid xarray indexing issues
        data_stim_ids = self.data['stimulus_id'].values
        data_id_to_idx = {sid: i for i, sid in enumerate(data_stim_ids)}
        y_indices = [data_id_to_idx[sid] for sid in pred_stimulus_ids]
        y_aligned = self.data.values[y_indices]
        
        # Get passage labels for the aligned data for GroupKFold
        # self.data has passage_label, so we can use y_indices to get them
        passage_labels_aligned = self.data['passage_label'].values[y_indices]
        
        # Define splits
        if self.topic_wise_cv:
            # Topic-Held-Out: Use GroupKFold with passage labels
            from sklearn.model_selection import GroupKFold
            n_splits = 10
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X_llm, y_aligned, groups=passage_labels_aligned))
            
            # DEBUG: Verify splits
            print(f"DEBUG: Topic-Wise CV Enabled. Unique groups: {len(set(passage_labels_aligned))}")
            print(f"DEBUG: Number of splits: {len(splits)}")
            train_idx, test_idx = splits[0]
            train_groups = set(passage_labels_aligned[train_idx])
            test_groups = set(passage_labels_aligned[test_idx])
            print(f"DEBUG: Split 0 - Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            print(f"DEBUG: Split 0 - Overlap between Train/Test groups: {train_groups.intersection(test_groups)}")
            assert len(train_groups.intersection(test_groups)) == 0, "CRITICAL: Group leakage detected!"
        else:
            # Random CV: Use KFold with shuffling
            from sklearn.model_selection import KFold
            n_splits = 10
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(X_llm, y_aligned))
        
        # Compute score
        score, diagnostics = self.metric(
            X_obj=X_obj_aligned,
            X_llm=X_llm,
            y=y_aligned,
            splits=splits
        )
        
        # Calculate Normalized Correlation
        original_alignment_score = diagnostics.get('original_alignment_score', None)
        objective_alignment_score = diagnostics.get('objective_alignment_score', None)
        
        original_normalized_alignment_score = None
        objective_normalized_alignment_score = None
        normalized_partial_r2 = None
        
        if self.ceiling is not None:
            # Normalize LLM Correlation
            if original_alignment_score is not None:
                try:
                    # Create a scalar Score object for normalization
                    raw_score = Score(original_alignment_score)
                    normalized_score = ceiling_normalize(raw_score, self.ceiling)
                    original_normalized_alignment_score = float(normalized_score.values)
                except Exception as e:
                    print(f"Warning: Failed to normalize LLM correlation: {e}")
                    original_normalized_alignment_score = None

            # Normalize Objective Correlation
            if objective_alignment_score is not None:
                try:
                    raw_score = Score(objective_alignment_score)
                    normalized_score = ceiling_normalize(raw_score, self.ceiling)
                    objective_normalized_alignment_score = float(normalized_score.values)
                except Exception as e:
                    print(f"Warning: Failed to normalize Objective correlation: {e}")
                    objective_normalized_alignment_score = None
            
            # Normalize Partial R2
            # Assuming ceiling is Pearson r, ceiling explained variance is r^2
            # We aggregate ceiling across neuroids (median/mean) to get a scalar ceiling R2?
            # Or we normalize per neuroid and then aggregate?
            # The metric returns aggregated score.
            # Let's try to normalize the aggregated score by the aggregated ceiling R2.
            try:
                # Ceiling is per neuroid.
                # We need to align ceiling to the neuroids we have (y_aligned is (n_samples, n_neuroids))
                # y_aligned comes from self.data.
                # self.ceiling should match self.data neuroids.
                # Let's check if we can get a scalar ceiling R2.
                ceiling_values = self.ceiling.values # (n_neuroids,)
                # If ceiling is r, r^2 is explained variance ceiling.
                ceiling_r2 = ceiling_values ** 2
                # Take median ceiling R2 (consistent with median aggregation in metric)
                median_ceiling_r2 = np.median(ceiling_r2)
                
                if median_ceiling_r2 > 0:
                    normalized_partial_r2 = score / median_ceiling_r2
                else:
                    normalized_partial_r2 = 0.0
            except Exception as e:
                print(f"Warning: Failed to normalize Partial R2: {e}")
                normalized_partial_r2 = None

        
        # Wrap in Score object
        final_score = Score(score)
        final_score.attrs['diagnostics'] = diagnostics
        final_score.attrs['model_identifier'] = candidate.identifier
        final_score.attrs['benchmark_identifier'] = self.identifier
        
        if original_alignment_score is not None:
            final_score.attrs['original_alignment_score'] = original_alignment_score
        if original_normalized_alignment_score is not None:
            final_score.attrs['original_normalized_alignment_score'] = original_normalized_alignment_score
            
        if objective_alignment_score is not None:
            final_score.attrs['objective_alignment_score'] = objective_alignment_score
        if objective_normalized_alignment_score is not None:
            final_score.attrs['objective_normalized_alignment_score'] = objective_normalized_alignment_score
            
        if normalized_partial_r2 is not None:
            final_score.attrs['normalized_partial_r2'] = normalized_partial_r2
        
        return final_score
