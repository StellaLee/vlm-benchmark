"""VL-Uncertainty (arXiv 2411.11919) — uncertainty via visual perturbation.

Idea: a confident VLM answer is robust to small, semantics-preserving changes to
the image; a hallucinated one is not. We render the question against several
Gaussian-blurred versions of the images, then measure the *semantic entropy* of
the resulting answers. Low entropy (answers agree across perturbations) -> high
confidence; high entropy -> likely hallucination.

confidence = 1 - normalized_semantic_entropy, in [0, 1].

Simplifications vs. the paper (documented intentionally):
- Clustering is by normalized-text exact match, a lightweight proxy for the
  paper's bidirectional-entailment ("semantic") clustering. Fine for MCQ /
  short answers; crude for long open-ended ones.
- We perturb the visual input only; the paper also perturbs the prompt (LLM
  side). A prompt-rephrasing variant can be layered on later.

Cost: len(blur_radii) API calls per sample (default 5).
"""

import asyncio
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from typing import List

from avbench.inference.client import VLMClient
from avbench.inference.parsing import extract_answer, is_abstention
from avbench.inference.strategies.base import (
    PromptStrategy, answer_instruction, register, render_question)
from avbench.schema import ImageRef, Prediction, Sample


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


@register("vl_uncertainty")
class VLUncertainty(PromptStrategy):
    def __init__(self, blur_radii=(0.0, 1.0, 2.0, 3.0, 4.0), temperature: float = 0.0):
        self.blur_radii = list(blur_radii)
        self.temperature = temperature

    def build_prompt(self, sample: Sample) -> str:
        return "{}\n\n{}\nAnswer:".format(render_question(sample), answer_instruction(sample))

    def _blur(self, images: List[ImageRef], radius: float, tmpdir: str) -> List[ImageRef]:
        if radius <= 0:
            return images  # radius 0 = the original images
        from PIL import Image, ImageFilter

        out = []
        for i, im in enumerate(images):
            try:
                img = Image.open(im.path).convert("RGB").filter(ImageFilter.GaussianBlur(radius))
                p = os.path.join(tmpdir, "r{}_{}.jpg".format(radius, i))
                img.save(p, "JPEG")
                out.append(ImageRef(path=p, camera=im.camera, frame_idx=im.frame_idx))
            except OSError:
                out.append(im)  # fall back to original if unreadable
        return out

    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        prompt = self.prompt_for(sample)
        base_imgs = self.images_for(sample)
        tmpdir = tempfile.mkdtemp(prefix="vlu_")
        try:
            variants = [self._blur(base_imgs, r, tmpdir) for r in self.blur_radii]
            results = await asyncio.gather(*[
                client.generate(prompt, imgs, n=1, temperature=self.temperature)
                for imgs in variants
            ])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        answers = [r[0].text for r in results]
        parsed = [extract_answer(t, sample) for t in answers]
        norm = [_normalize(a) for a in parsed if a]

        confidence, entropy, top_answer = None, None, None
        if norm:
            counts = Counter(norm)
            n = sum(counts.values())
            probs = [c / n for c in counts.values()]
            entropy = -sum(p * math.log(p) for p in probs)
            h_max = math.log(len(self.blur_radii)) if len(self.blur_radii) > 1 else 1.0
            confidence = 1.0 - (entropy / h_max if h_max > 0 else 0.0)
            top_norm = counts.most_common(1)[0][0]
            top_answer = next((a for a in parsed if _normalize(a) == top_norm), None)

        return Prediction(
            sample_id=sample.sample_id,
            model=client.model,
            strategy=self.name,
            raw_text=answers[0] if answers else "",
            answer=top_answer,
            verbal_confidence=confidence,  # 1 - normalized semantic entropy
            samples=parsed,
            abstained=is_abstention(answers[0]) if answers else False,
            usage={"blur_radii": self.blur_radii, "n_perturb": len(self.blur_radii),
                   "semantic_entropy": round(entropy, 4) if entropy is not None else None,
                   "backend": results[0][0].usage.get("backend") if results else None},
            condition=self.condition(),
        )
