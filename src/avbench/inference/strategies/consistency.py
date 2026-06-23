"""Self-consistency confidence (Wang et al. 2022 applied to confidence).

Samples the model k times at non-zero temperature; the agreement rate of the
majority answer serves as the confidence signal. No logprobs or verbal numbers
needed, so it works on any backend.
"""

from collections import Counter

from avbench.inference.client import VLMClient
from avbench.inference.parsing import extract_answer, is_abstention
from avbench.inference.strategies.base import PromptStrategy, register, render_question
from avbench.schema import Prediction, Sample


@register("consistency")
class Consistency(PromptStrategy):
    def __init__(self, k: int = 5, temperature: float = 0.7):
        self.k = k
        self.temperature = temperature

    def build_prompt(self, sample: Sample) -> str:
        how = (
            "Answer with the single best option letter."
            if sample.options
            else "Answer concisely."
        )
        return "{}\n\n{}\nAnswer:".format(render_question(sample), how)

    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        results = await client.generate(
            self.build_prompt(sample), sample.images, n=self.k, temperature=self.temperature
        )
        answers = [extract_answer(r.text, sample) for r in results]
        counts = Counter(a for a in answers if a)
        if counts:
            top, n_top = counts.most_common(1)[0]
        else:
            top, n_top = None, 0
        confidence = n_top / float(len(answers)) if answers else None
        return Prediction(
            sample_id=sample.sample_id,
            model=client.model,
            strategy=self.name,
            raw_text=results[0].text if results else "",
            answer=top,
            verbal_confidence=confidence,  # agreement rate as confidence
            samples=answers,
            abstained=bool(results) and is_abstention(results[0].text),
            usage={"k": self.k, "temperature": self.temperature,
                   "backend": results[0].usage.get("backend") if results else None},
        )
