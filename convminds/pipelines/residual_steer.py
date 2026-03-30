from __future__ import annotations

from typing import Any, Callable

import torch

import convminds as cm
from convminds.data import BrainDataModule
from convminds.models.residual_steer import ResidualSteerLM
from convminds.subjects import HFArtificialSubject, HumanSubject


class _BoundPenalizedLoss:
    """
    Helper loss function that attaches to the model's computed penalty dynamically.
    """
    def __init__(self, model: ResidualSteerLM, lambda_weight: float = 1.0):
        self.model = model
        self.loss_fn = cm.objectives.PenalizedCrossEntropy(lambda_weight=lambda_weight)
        
    def __call__(self, outputs, labels):
        penalty = self.model.get_penalty()
        return self.loss_fn(outputs, labels, penalty)


class ResidualSteerPipeline:
    """
    End-to-end replication pipeline for the ResidualSteerLM.
    Two-stage training: MSE warm-up on the encoder, then end-to-end generation training
    with the localized injection norm penalty.
    """

    def __init__(
        self,
        *,
        benchmark=None,
        subject_id: int | str = 1,
        llm_id: str = "meta-llama/Llama-2-7b-hf",
        batch_size: int = 16,
        tokenizer: Any | None = None,
        injection_layer: int = 16,
        lambda_weight: float = 1.0,
    ):
        if benchmark is None:
            raise ValueError("ResidualSteerPipeline requires an explicit benchmark instance.")

        self.subject_id = subject_id
        self.llm_id = llm_id
        self.tokenizer = tokenizer

        self.human = HumanSubject(subject_ids=[str(subject_id)])
        self.oracle = HFArtificialSubject(llm_id, layers=[-1])

        # Phase 1: Structural Setup (incorporates Windowing and Dimensionality Reduction safely)
        self.datamodule = BrainDataModule(
            benchmark=benchmark,
            human_subject=self.human,
            artificial_subject=self.oracle,
            stateless_transforms=[cm.transforms.HRFWindow(t=4)],
            stateful_transforms=[cm.transforms.PCA(n_components=1000)],
            batch_size=batch_size,
            text_transform=self._build_text_transform(tokenizer) if tokenizer is not None else None,
        )

        self.model = ResidualSteerLM(
            llm_id=llm_id, 
            encoder_in_dim=1000, 
            injection_layer=injection_layer, 
            num_frames=4
        )
        self.lambda_weight = lambda_weight

    def train(self, *, align_epochs: int = 5, gen_epochs: int = 10) -> None:
        self.datamodule.setup()
        train_loader = self.datamodule.train_dataloader()

        # Phase 1 Training: MSE "Warm-Up" on the Brain Encoder
        align_trainer = cm.trainers.GradientTrainer(
            model=self.model.encoder,
            loss_fn=torch.nn.MSELoss(),
            lr=1e-3,
        )
        align_trainer.fit(train_loader, target_key="target_latents", epochs=align_epochs)

        if self.tokenizer is None:
            raise ValueError("Generative tuning requires a tokenizer to produce text_input_ids.")

        # Phase 3 Training: End-to-end generation with Norm Penalty
        bound_loss = _BoundPenalizedLoss(self.model, lambda_weight=self.lambda_weight)
        
        gen_trainer = cm.trainers.GradientTrainer(
            model=self.model,
            loss_fn=bound_loss,
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
                brain_tensor=brain_tensor,
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
