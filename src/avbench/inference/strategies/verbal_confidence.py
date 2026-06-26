"""Verbal-confidence elicitation (Lin et al. 2022; Tian et al. 2023).

Asks the model to answer and then state a numeric confidence. This is the primary
signal for the prototype's ECE/AUROC, since it works on API models that hide
token logprobs.
"""

from avbench.inference.client import VLMClient
from avbench.inference.parsing import extract_answer, extract_confidence, is_abstention
from avbench.inference.strategies.base import (
    PromptStrategy, answer_instruction, register, render_question)
from avbench.schema import Prediction, Sample

_TEMPLATE = (
    "{body}\n\n"
    "{how_to_answer}\n"
    "Then, on a new line, give your confidence that the answer is correct as a "
    "percentage from 0 to 100.\n\n"
    "Respond in exactly this format:\n"
    "Answer: <your answer>\n"
    "Confidence: <0-100>"
)


@register("verbal_confidence")
class VerbalConfidence(PromptStrategy):
    def build_prompt(self, sample: Sample) -> str:
        return _TEMPLATE.format(body=render_question(sample), how_to_answer=answer_instruction(sample))

    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        imgs = self.images_for(sample)
        res = (await client.generate(self.prompt_for(sample), imgs, n=1))[0]
        return Prediction(
            sample_id=sample.sample_id,
            model=client.model,
            strategy=self.name,
            raw_text=res.text,
            answer=extract_answer(res.text, sample),
            verbal_confidence=extract_confidence(res.text),
            token_logprob=res.avg_logprob,
            abstained=is_abstention(res.text),
            usage=res.usage,
            condition=self.condition(),
        )
