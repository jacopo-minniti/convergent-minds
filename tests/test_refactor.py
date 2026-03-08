import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

import convminds
from convminds import HumanSubject, InMemoryBenchmark, LinearDecoder
from convminds.metrics import linear_r2
from convminds.pipelines import run_basic_decoder_pipeline
from convminds.subjects import ArtificialSubject


def _latent_for_text(text: str, dim: int = 10) -> np.ndarray:
    seed = abs(hash(text)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim)


def build_synthetic_benchmark(*, cv: int = 1, topic_splitting: bool = True) -> InMemoryBenchmark:
    rng = np.random.default_rng(123)
    records = []
    latents = []
    duplicate = "Shared test sentence."
    for topic_index in range(6):
        topic = f"topic-{topic_index:02d}"
        for sentence_index in range(3):
            text = duplicate if topic_index % 2 == 0 and sentence_index == 0 else f"{topic} sentence {sentence_index}"
            latent = _latent_for_text(text)
            records.append(
                {
                    "stimulus_id": f"stim-{len(records):03d}",
                    "text": text,
                    "topic": topic,
                    "metadata": {"latent": latent.tolist(), "topic_index": topic_index},
                }
            )
            latents.append(latent)

    latents = np.asarray(latents, dtype=float)
    human_values = latents @ rng.normal(size=(latents.shape[1], 12))
    return InMemoryBenchmark(
        identifier="test.synthetic",
        stimuli=records,
        human_values=human_values,
        feature_ids=[f"neuroid-{index}" for index in range(human_values.shape[1])],
        cv=cv,
        topic_splitting=topic_splitting,
        storage_mode="glm",
        recording_type="synthetic-fmri",
    )


class MockArtificialSubject(ArtificialSubject):
    def __init__(self, feature_dim: int = 14):
        super().__init__()
        self.feature_dim = feature_dim
        self._projection = None

    def identifier(self) -> str:
        return "mock-artificial"

    def subject_config(self) -> dict:
        return {"kind": "mock-artificial", "feature_dim": self.feature_dim}

    def record(self, benchmark, **kwargs):
        rng = np.random.default_rng(999)
        latent_dim = len(benchmark.stimuli[0].metadata["latent"])
        if self._projection is None:
            self._projection = rng.normal(size=(latent_dim, self.feature_dim))

        values = []
        stimulus_ids = []
        for index in reversed(range(len(benchmark.stimuli))):
            record = benchmark.stimuli[index]
            latent = np.asarray(record.metadata["latent"], dtype=float)
            values.append(latent @ self._projection)
            stimulus_ids.append(record.stimulus_id)

        self._store_recordings(
            benchmark,
            np.asarray(values, dtype=float),
            stimulus_ids=stimulus_ids,
            feature_ids=[f"mock-{index}" for index in range(self.feature_dim)],
            metadata={"source": "mock"},
        )
        return self


class RefactorWorkflowTests(unittest.TestCase):
    def test_single_split_alignment_and_leakage_protection(self):
        benchmark = build_synthetic_benchmark(cv=1, topic_splitting=True)
        humans = HumanSubject()
        artificial = MockArtificialSubject()

        humans.record(benchmark)
        artificial.record(benchmark)

        self.assertEqual(len(humans.neurons), 1)
        self.assertEqual(len(artificial.neurons), 1)

        human_train_ids = [record.stimulus_id for record in humans.split_stimuli[0]["train"]]
        artificial_train_ids = [record.stimulus_id for record in artificial.split_stimuli[0]["train"]]
        self.assertEqual(human_train_ids, artificial_train_ids)

        train_texts = {record.text for record in humans.split_stimuli[0]["train"]}
        test_texts = {record.text for record in humans.split_stimuli[0]["test"]}
        self.assertTrue(train_texts.isdisjoint(test_texts))

        train_topics = {record.topic for record in humans.split_stimuli[0]["train"]}
        test_topics = {record.topic for record in humans.split_stimuli[0]["test"]}
        self.assertTrue(train_topics.isdisjoint(test_topics))

    def test_pipeline_and_cached_score_display(self):
        with tempfile.TemporaryDirectory() as tempdir:
            original_home = os.environ.get("CONVMINDS_HOME")
            os.environ["CONVMINDS_HOME"] = tempdir
            try:
                benchmark = build_synthetic_benchmark(cv=3, topic_splitting=False)
                humans = HumanSubject()
                artificial = MockArtificialSubject()
                decoder = LinearDecoder(loss="mse", l2_penalty=0.01)

                result = run_basic_decoder_pipeline(
                    artificial,
                    humans,
                    benchmark,
                    decoder,
                    save_score=True,
                )

                self.assertEqual(len(result.split_scores), 3)
                self.assertGreater(result.mean_score, -10.0)
                self.assertIsNotNone(result.cache_path)
                self.assertTrue(os.path.exists(result.cache_path))

                decoder.reset().train(humans.neurons[0]["train"], artificial.neurons[0]["train"])
                score = linear_r2(decoder, humans.neurons[0]["test"], artificial.neurons[0]["test"])
                self.assertIsInstance(score.value, float)

                stream = io.StringIO()
                with redirect_stdout(stream):
                    manifest = convminds.display_score(
                        artificial_subject=artificial.identifier(),
                        benchmark=benchmark.identifier,
                    )
                self.assertIsNotNone(manifest)
                self.assertIn("Cached score", stream.getvalue())
            finally:
                if original_home is None:
                    os.environ.pop("CONVMINDS_HOME", None)
                else:
                    os.environ["CONVMINDS_HOME"] = original_home


if __name__ == "__main__":
    unittest.main()
