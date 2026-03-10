from __future__ import annotations

from typing import Any, Callable

import torch

import convminds as cm
from convminds.data import BrainDataModule
from convminds.models import SpatialPrefixLM
from convminds.subjects import HFArtificialSubject, HumanSubject


class MindLLM:
    """
    End-to-end replication of MindLLM (Qiu et al.) with two-stage training.
    """

    def __init__(
        self,
        *,
        benchmark=None,
        subject_id: int | str = 1,
        llm_id: str = "lmsys/vicuna-7b-v1.5",
        batch_size: int = 16,
        tokenizer: Any | None = None,
    ):
        if benchmark is None:
            raise ValueError("MindLLM requires an explicit benchmark instance for data access.")

        self.subject_id = subject_id
        self.llm_id = llm_id
        self.tokenizer = tokenizer

        self.human = HumanSubject(subject_ids=[str(subject_id)])
        self.oracle = HFArtificialSubject(llm_id, layers=[-1])

        self.datamodule = BrainDataModule(
            benchmark=benchmark,
            human_subject=self.human,
            artificial_subject=self.oracle,
            stateful_transforms=cm.transforms.ZScore(dim="time"),
            batch_size=batch_size,
            text_transform=self._build_text_transform(tokenizer) if tokenizer is not None else None,
        )

        self.model = SpatialPrefixLM(llm_id=llm_id, num_queries=128, llm_dim=4096)

    def train(self, *, align_epochs: int = 5, gen_epochs: int = 10) -> None:
        self.datamodule.setup()
        train_loader = self.datamodule.train_dataloader()

        align_trainer = cm.trainers.GradientTrainer(
            model=self.model.encoder,
            loss_fn=torch.nn.MSELoss(),
            lr=1e-3,
        )
        align_trainer.fit(train_loader, target_key="target_latents", epochs=align_epochs)

        if self.tokenizer is None:
            raise ValueError("Generative tuning requires a tokenizer to produce text_input_ids.")

        gen_trainer = cm.trainers.GradientTrainer(
            model=self.model,
            loss_fn=cm.objectives.NextTokenCrossEntropy(),
            lr=1e-4,
        )
        gen_trainer.fit(train_loader, target_key="labels", epochs=gen_epochs)

    def evaluate(self) -> float:
        if self.tokenizer is None:
            raise ValueError("Evaluation requires a tokenizer to decode generated text.")

        predictions: list[str] = []
        references: list[str] = []
        dataloader = self.datamodule.test_dataloader()
        for batch in dataloader:
            brain_tensor = batch["brain_tensor"]
            input_ids = batch.get("text_input_ids")
            attention_mask = batch.get("attention_mask")
            generated = self.model.generate(
                brain_tensor,
                text_input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
            )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(decoded)
            references.extend(batch["text"])

        evaluator = cm.metrics.text.BLEU(ngram=4)
        return evaluator.compute(predictions, references)

    def _build_text_transform(self, tokenizer) -> Callable[[str], dict[str, torch.Tensor]]:
        def encode(text: str) -> dict[str, torch.Tensor]:
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = encoded["input_ids"].squeeze(0)
            payload = {
                "text_input_ids": input_ids,
                "labels": input_ids.clone(),
            }
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                payload["attention_mask"] = attention_mask.squeeze(0)
            return payload

        return encode
